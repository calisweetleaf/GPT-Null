"""
Asynchronous multimodal tokenizer multiplexer for GPT transformers.
Handles multi-modal inputs with async preprocessing, buffering, streaming, and source flags.
Returns token embeddings + metadata for downstream fusion.

Author: Morpheus
Date: 2025-05-03
Version: 0.1.0
License: MIT
Copyright (c) 2025 Morpheus
"""

import logging
import time
import concurrent.futures
import torch
import numpy as np
import os
import json
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, TypeVar, Generic, cast, Generator
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
import threading
from contextlib import contextmanager
from tokenizers import SentencePieceBPETokenizer
import traceback
import hashlib
import prometheus_client as prom
import socket
from pathlib import Path

# Configure logging with structured output
logger = logging.getLogger("gpt4o.tokenizer_mux")

# Define metrics
TOKENIZATION_LATENCY = prom.Summary('tokenization_latency_seconds', 
           'Time spent tokenizing inputs', 
           ['modality'])
TOKENIZATION_ERRORS = prom.Counter('tokenization_errors_total', 
          'Total tokenization errors', 
          ['modality', 'error_type'])
CACHE_HITS = prom.Counter('tokenizer_cache_hits_total', 
       'Total tokenizer cache hits', 
       ['modality'])
TOKENS_PROCESSED = prom.Counter('tokens_processed_total', 
          'Total tokens processed', 
          ['modality'])
QUEUE_DEPTH = prom.Gauge('tokenizer_queue_depth', 
      'Current pre-tokenization buffer depth', 
      ['modality'])
CIRCUIT_BREAKER_STATE = prom.Enum('tokenizer_circuit_breaker_state',
         'Circuit breaker state',
         ['modality'],
         states=['closed', 'open', 'half_open'])

# Type variables for generics
T = TypeVar('T')
U = TypeVar('U')

class TokenizerErrorCode(Enum):
 """Error codes for tokenizer operations."""
 INVALID_INPUT = auto()
 PREPROCESSING_ERROR = auto()
 TOKENIZATION_ERROR = auto()
 RESOURCE_ERROR = auto()
 CONFIG_ERROR = auto()
 TIMEOUT_ERROR = auto()
 EXTERNAL_SERVICE_ERROR = auto()
 UNKNOWN_ERROR = auto()

class TokenizerError(Exception):
 """Base exception for tokenizer errors with error code and context."""
 
 def __init__(self, message: str, code: TokenizerErrorCode, details: Optional[Dict[str, Any]] = None):
  """
  Initialize tokenizer error with message, code and details.
  
  Args:
   message: Human-readable error message
   code: Error code from TokenizerErrorCode enum
   details: Optional dictionary with error context
  """
  self.code = code
  self.details = details or {}
  super().__init__(message)

class PreprocessingError(TokenizerError):
 """Raised when preprocessing of inputs fails."""
 def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
  super().__init__(message, TokenizerErrorCode.PREPROCESSING_ERROR, details)

class TokenizationError(TokenizerError):
 """Raised when tokenization of preprocessed inputs fails."""
 def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
  super().__init__(message, TokenizerErrorCode.TOKENIZATION_ERROR, details)

class InvalidInputError(TokenizerError):
 """Raised when input validation fails."""
 def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
  super().__init__(message, TokenizerErrorCode.INVALID_INPUT, details)

class ResourceError(TokenizerError):
 """Raised when resource allocation or management fails."""
 def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
  super().__init__(message, TokenizerErrorCode.RESOURCE_ERROR, details)

class ConfigError(TokenizerError):
 """Raised when configuration validation fails."""
 def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
  super().__init__(message, TokenizerErrorCode.CONFIG_ERROR, details)

class TimeoutError(TokenizerError):
 """Raised when an operation times out."""
 def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
  super().__init__(message, TokenizerErrorCode.TIMEOUT_ERROR, details)

class ExternalServiceError(TokenizerError):
 """Raised when an external service call fails."""
 def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
  super().__init__(message, TokenizerErrorCode.EXTERNAL_SERVICE_ERROR, details)

class ModalityType(Enum):
    """Supported data modalities."""
    TEXT = "text"
    STRUCTURED = "structured"  # For code, JSON, YAML, etc.
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TOOL = "tool"
    EMBEDDING = "embedding"
    LIVE_WEB = "live_web"
    LIDAR = "lidar"
    GPS = "gps"
    CLOCK = "clock"
    RM_RF = "rm_rf"  # Removal/deletion operations
    ADS_B = "ads_b"  # Aircraft tracking data
    EYES = "eyes"  # Structured data from the vision system
    EARS = "ears"  # Structured data from the audio system
    SPATIAL = "spatial" # Coordinated spatial sensor data

class CircuitBreakerState(Enum):
 """Circuit breaker states."""
 CLOSED = "closed"  # Normal operation
 OPEN = "open"      # Failing, rejecting requests
 HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
 """Circuit breaker pattern implementation for external service calls."""
 
 def __init__(self, name: str, max_failures: int = 3, reset_timeout: float = 30.0):
  """
  Initialize circuit breaker.
  
  Args:
   name: Name identifier for the circuit breaker
   max_failures: Maximum number of consecutive failures before opening circuit
   reset_timeout: Time in seconds to wait before attempting reset
  """
  self.name = name
  self.max_failures = max_failures
  self.reset_timeout = reset_timeout
  self.failures = 0
  self.state = CircuitBreakerState.CLOSED
  self.last_failure_time = 0
  self._lock = threading.RLock()
  CIRCUIT_BREAKER_STATE.labels(modality=name).state(self.state.value)
 
 def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
  """
  Decorator for protecting functions with circuit breaker.
  
  Args:
   func: Function to protect with circuit breaker
   
  Returns:
   Wrapped function with circuit breaker protection
  """
  @wraps(func)
  def wrapper(*args: Any, **kwargs: Any) -> T:
   with self._lock:
    if self.state == CircuitBreakerState.OPEN:
     if time.time() - self.last_failure_time > self.reset_timeout:
      logger.info(f"Circuit {self.name} half-open, attempting reset")
      self.state = CircuitBreakerState.HALF_OPEN
      CIRCUIT_BREAKER_STATE.labels(modality=self.name).state(self.state.value)
     else:
      raise ResourceError(
       f"Circuit breaker {self.name} open, call rejected",
       {"state": self.state.value, "failures": self.failures}
      )
    
    try:
     result = func(*args, **kwargs)
     
     if self.state == CircuitBreakerState.HALF_OPEN:
      logger.info(f"Circuit {self.name} reset successful, closing circuit")
      self.failures = 0
      self.state = CircuitBreakerState.CLOSED
      CIRCUIT_BREAKER_STATE.labels(modality=self.name).state(self.state.value)
     
     return result
     
    except Exception as e:
     self.failures += 1
     self.last_failure_time = time.time()
     
     if self.failures >= self.max_failures:
      logger.warning(f"Circuit {self.name} opening after {self.failures} failures")
      self.state = CircuitBreakerState.OPEN
      CIRCUIT_BREAKER_STATE.labels(modality=self.name).state(self.state.value)
     
     raise e
  
  return wrapper

@dataclass
class TokenizerConfig:
 """Configuration for the tokenizer system with validation."""
 max_text_length: int = 4096
 max_image_pixels: Tuple[int, int] = (768, 768)
 audio_sample_rate: int = 24000
 enable_caching: bool = True
 cache_size: int = 10000
 num_threads: int = 4
 batch_size: int = 32
 vqvae_codebook_size: int = 65536
 vqvae_embedding_dim: int = 32
 audio_codebooks: int = 4
 audio_codebook_size: int = 65536
 video_frame_rate: int = 4
 pre_tokenization_buffer: int = 128
 timeout_ms: int = 5000
 circuit_breaker_max_failures: int = 3
 circuit_breaker_reset_sec: float = 30.0
 model_paths: Dict[str, str] = field(default_factory=lambda: {
  "text": os.environ.get("TEXT_MODEL_PATH", "./models/gpt4o_text.model"),
  "image": os.environ.get("IMAGE_MODEL_PATH", "./models/gpt4o_image.model"),
  "audio": os.environ.get("AUDIO_MODEL_PATH", "./models/gpt4o_audio.model"),
  "video": os.environ.get("VIDEO_MODEL_PATH", "./models/gpt4o_video.model"),
 })
 
 def __post_init__(self) -> None:
  """Validate configuration after initialization."""
  self._validate_config()
  
  # Load from environment variables if available
  if os.environ.get("MAX_TEXT_LENGTH"):
   self.max_text_length = int(os.environ.get("MAX_TEXT_LENGTH", self.max_text_length))
  if os.environ.get("CACHE_SIZE"):
   self.cache_size = int(os.environ.get("CACHE_SIZE", self.cache_size))
  if os.environ.get("NUM_THREADS"):
   self.num_threads = int(os.environ.get("NUM_THREADS", self.num_threads))
  if os.environ.get("ENABLE_CACHING") is not None:
   self.enable_caching = os.environ.get("ENABLE_CACHING", "true").lower() == "true"
 
 def _validate_config(self) -> None:
  """Validate configuration parameters."""
  if self.max_text_length <= 0:
   raise ConfigError("max_text_length must be positive", {"value": self.max_text_length})
  
  if self.max_image_pixels[0] <= 0 or self.max_image_pixels[1] <= 0:
   raise ConfigError("image dimensions must be positive", {"value": self.max_image_pixels})
   
  if self.audio_sample_rate <= 0:
   raise ConfigError("audio_sample_rate must be positive", {"value": self.audio_sample_rate})
   
  if self.num_threads <= 0:
   raise ConfigError("num_threads must be positive", {"value": self.num_threads})
   
  if self.batch_size <= 0:
   raise ConfigError("batch_size must be positive", {"value": self.batch_size})
   
  if self.cache_size < 0:
   raise ConfigError("cache_size cannot be negative", {"value": self.cache_size})
   
  if self.pre_tokenization_buffer <= 0:
   raise ConfigError("pre_tokenization_buffer must be positive", {"value": self.pre_tokenization_buffer})
 
 def to_dict(self) -> Dict[str, Any]:
  """Convert configuration to dictionary."""
  return asdict(self)
 
 @classmethod
 def from_dict(cls, config_dict: Dict[str, Any]) -> 'TokenizerConfig':
  """Create configuration from dictionary."""
  return cls(**config_dict)
 
 @classmethod
 def from_json(cls, json_path: str) -> 'TokenizerConfig':
  """Load configuration from JSON file."""
  try:
   with open(json_path, 'r') as f:
    config_dict = json.load(f)
   return cls.from_dict(config_dict)
  except (IOError, json.JSONDecodeError) as e:
   raise ConfigError(f"Failed to load config from {json_path}", {"error": str(e)})

@dataclass
class TokenizationResult:
 """Result of tokenization with metrics and metadata."""
 tokens: torch.Tensor
 attention_mask: torch.Tensor
 modality: ModalityType
 source_shape: Tuple[int, ...]
 processing_time_ms: float
 cached: bool = False
 source_flag: Optional[torch.Tensor] = None
 metadata: Optional[Dict[str, Any]] = None
 
 def __post_init__(self) -> None:
  """Validate TokenizationResult after initialization."""
  if self.metadata is None:
   self.metadata = {}
  
  # Ensure all tensors are on the same device
  if self.source_flag is not None and self.tokens.device != self.source_flag.device:
   self.source_flag = self.source_flag.to(self.tokens.device)
 
 def to_dict(self) -> Dict[str, Any]:
  """Convert result to dictionary for serialization."""
  return {
   "tokens_shape": list(self.tokens.shape),
   "attention_mask_shape": list(self.attention_mask.shape),
   "modality": self.modality.value,
   "source_shape": list(self.source_shape),
   "processing_time_ms": self.processing_time_ms,
   "cached": self.cached,
   "metadata": self.metadata
  }
 
 def to_device(self, device: torch.device) -> 'TokenizationResult':
  """Move tensors to specified device."""
  return TokenizationResult(
   tokens=self.tokens.to(device),
   attention_mask=self.attention_mask.to(device),
   modality=self.modality,
   source_shape=self.source_shape,
   processing_time_ms=self.processing_time_ms,
   cached=self.cached,
   source_flag=self.source_flag.to(device) if self.source_flag is not None else None,
   metadata=self.metadata
  )

class ModalityProcessor:
 """Base class for modality-specific processors."""
 
 def __init__(self, config: TokenizerConfig):
  """
  Initialize the modality processor.
  
  Args:
   config: Tokenizer configuration
  """
  self.config = config
  self.circuit_breaker = CircuitBreaker(
   name=self.__class__.__name__,
   max_failures=config.circuit_breaker_max_failures,
   reset_timeout=config.circuit_breaker_reset_sec
  )
  
 async def preprocess(self, inputs: Any) -> Any:
  """
  Preprocess inputs asynchronously.
  
  Args:
   inputs: Raw inputs for the modality
   
  Returns:
   Preprocessed inputs ready for tokenization
   
  Raises:
   PreprocessingError: When preprocessing fails
  """
  raise NotImplementedError("Subclasses must implement preprocess method")
 
 @CircuitBreaker(name="tokenize")
 def tokenize(self, inputs: Any) -> TokenizationResult:
  """
  Tokenize preprocessed inputs.
  
  Args:
   inputs: Preprocessed inputs
   
  Returns:
   TokenizationResult with tokens and metadata
   
  Raises:
   TokenizationError: When tokenization fails
  """
  raise NotImplementedError("Subclasses must implement tokenize method")
 
 @contextmanager
 def _timed_operation(self, operation_name: str) -> Generator[None, None, None]:
  """
  Context manager for timing operations.
  
  Args:
   operation_name: Name of the operation being timed
   
  Yields:
   None
  """
  start_time = time.perf_counter()
  try:
   yield
  finally:
   end_time = time.perf_counter()
   duration_ms = (end_time - start_time) * 1000
   logger.debug(f"{operation_name} took {duration_ms:.2f}ms")

class TextProcessor(ModalityProcessor):
 """Processor for text inputs."""
 
 def __init__(self, config: TokenizerConfig):
  """
  Initialize text processor with tokenizer model.
  
  Args:
   config: Tokenizer configuration
  """
  super().__init__(config)
  try:
   model_path = config.model_paths["text"]
   if os.path.exists(model_path):
    self.tokenizer = SentencePieceBPETokenizer(model_path)
   else:
    # Fallback to default model
    self.tokenizer = SentencePieceBPETokenizer()
    logger.warning(f"Text model not found at {model_path}. Using default model.")
  except Exception as e:
   logger.error(f"Failed to load text tokenizer: {str(e)}")
   self.tokenizer = SentencePieceBPETokenizer()

 async def preprocess(self, text: str) -> str:
  """
  Preprocess text input.
  
  Args:
   text: Raw text input
   
  Returns:
   Preprocessed text
   
  Raises:
   PreprocessingError: When text preprocessing fails
   InvalidInputError: When input is not valid text
  """
  if not isinstance(text, str):
   raise InvalidInputError("Text input must be a string", {"type": type(text).__name__})
  
  try:
   # Basic text preprocessing
   text = text.strip()
   
   # Truncate if needed
   if len(text) > self.config.max_text_length * 4:  # Approximate character limit
    logger.warning(f"Text input truncated from {len(text)} chars")
    text = text[:self.config.max_text_length * 4]
   
   return text
  except Exception as e:
   logger.error(f"Text preprocessing error: {str(e)}")
   raise PreprocessingError(f"Failed to preprocess text: {str(e)}", {"error": str(e)})
 
 @lru_cache(maxsize=10000)
 def _cached_tokenize(self, text: str) -> Dict[str, torch.Tensor]:
  """
  Cached tokenization for text inputs.
  
  Args:
   text: Preprocessed text
   
  Returns:
   Dictionary with input_ids and attention_mask
   
  Raises:
   TokenizationError: When tokenization fails
  """
  try:
   encoding = self.tokenizer.encode(text)
   ids = encoding.ids[:self.config.max_text_length]
   input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
   attn_mask = torch.ones_like(input_ids)
   return {"input_ids": input_ids, "attention_mask": attn_mask}
  except Exception as e:
   logger.error(f"Text tokenization error in cache: {str(e)}")
   raise TokenizationError(f"Failed to tokenize text: {str(e)}", {"error": str(e)})
 
 def tokenize(self, text: str) -> TokenizationResult:
  """
  Tokenize preprocessed text.
  
  Args:
   text: Preprocessed text input
   
  Returns:
   TokenizationResult with text tokens and metadata
   
  Raises:
   TokenizationError: When tokenization fails
  """
  with self._timed_operation("Text tokenization"):
   start_time = time.perf_counter()
   cached = False
   
   try:
    if self.config.enable_caching:
     outputs = self._cached_tokenize(text)
     cached = True
     if cached:
      CACHE_HITS.labels(modality="text").inc()
    else:
     encoding = self.tokenizer.encode(text)
     ids = encoding.ids[:self.config.max_text_length]
     input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
     attn_mask = torch.ones_like(input_ids)
     outputs = {"input_ids": input_ids, "attention_mask": attn_mask}
    
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    TOKENIZATION_LATENCY.labels(modality="text").observe(processing_time_ms / 1000)
    TOKENS_PROCESSED.labels(modality="text").inc(outputs["input_ids"].numel())
    
    return TokenizationResult(
     tokens=outputs["input_ids"],
     attention_mask=outputs["attention_mask"],
     modality=ModalityType.TEXT,
     source_shape=outputs["input_ids"].shape,
     processing_time_ms=processing_time_ms,
     cached=cached,
     source_flag=torch.zeros_like(outputs["input_ids"]),
     metadata={
      "type": "text",
      "length": outputs["input_ids"].size(1),
      "text_hash": hashlib.md5(text.encode()).hexdigest()[:8]
     }
    )
   except Exception as e:
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    logger.error(f"Text tokenization failed: {str(e)}")
    TOKENIZATION_ERRORS.labels(modality="text", error_type=type(e).__name__).inc()
    raise TokenizationError(f"Failed to tokenize text: {str(e)}", {"error": str(e)})

class ImageProcessor(ModalityProcessor):
 """Processor for image inputs."""
 
 def __init__(self, config: TokenizerConfig):
  """
  Initialize image processor with VQVAE model.
  
  Args:
   config: Tokenizer configuration
  """
  super().__init__(config)
  self.codebook_size = config.vqvae_codebook_size
  self.embedding_dim = config.vqvae_embedding_dim
  
  # In production, we'd load actual models here
  try:
   model_path = config.model_paths["image"]
   if not os.path.exists(model_path):
    logger.warning(f"Image model not found at {model_path}. Using mock implementation.")
  except Exception as e:
   logger.error(f"Failed to load image model: {str(e)}")

 async def preprocess(self, image: torch.Tensor) -> torch.Tensor:
  """
  Preprocess image input.
  
  Args:
   image: Raw image tensor of shape [C, H, W]
   
  Returns:
   Preprocessed image tensor
   
  Raises:
   PreprocessingError: When image preprocessing fails
   InvalidInputError: When input is not a valid image tensor
  """
  if not isinstance(image, torch.Tensor):
   raise InvalidInputError("Image input must be a torch.Tensor", {"type": type(image).__name__})
  
  if len(image.shape) != 3 or image.shape[0] not in [1, 3, 4]:
   raise InvalidInputError(
    "Image must have shape [C, H, W] with C=1,3,4", 
    {"shape": list(image.shape)}
   )
  
  try:
   # Resize image to configured dimensions
   return torch.nn.functional.interpolate(
    image.unsqueeze(0),
    size=self.config.max_image_pixels,
    mode='bilinear',
    align_corners=False
   ).squeeze(0)
  except Exception as e:
   logger.error(f"Image preprocessing error: {str(e)}")
   raise PreprocessingError(f"Failed to preprocess image: {str(e)}", {"error": str(e)})
 
 def tokenize(self, image: torch.Tensor) -> TokenizationResult:
  """
  Tokenize preprocessed image.
  
  Args:
   image: Preprocessed image tensor
   
  Returns:
   TokenizationResult with image tokens and metadata
   
  Raises:
   TokenizationError: When tokenization fails
  """
  with self._timed_operation("Image tokenization"):
   start_time = time.perf_counter()
   
   try:
    # In production, this would use a real VQVAE model
    # For now, we'll implement a deterministic tokenization based on image patches
    
    # Extract image properties
    c, h, w = image.shape
    
    # Calculate patch size for tokenization (16x16 patches are common)
    patch_size = 16
    h_patches = h // patch_size
    w_patches = w // patch_size
    
    # Create deterministic tokens based on image content
    # In a real implementation, this would be the output of a trained VQVAE
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patch_means = patches.mean(dim=[3, 4])
    
    # Convert patch statistics to token indices
    token_indices = (patch_means.sum(dim=0) * 1000).long() % self.codebook_size
    tokens = token_indices.reshape(1, -1)  # [1, h_patches * w_patches]
    
    # Create attention mask
    attn_mask = torch.ones_like(tokens)
    
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    TOKENIZATION_LATENCY.labels(modality="image").observe(processing_time_ms / 1000)
    TOKENS_PROCESSED.labels(modality="image").inc(tokens.numel())
    
    return TokenizationResult(
     tokens=tokens,
     attention_mask=attn_mask,
     modality=ModalityType.IMAGE,
     source_shape=image.shape,
     processing_time_ms=processing_time_ms,
     cached=False,
     source_flag=torch.zeros_like(tokens),
     metadata={
      "type": "image",
      "height": h,
      "width": w,
      "channels": c,
      "patch_size": patch_size,
      "patches": (h_patches, w_patches)
     }
    )
   except Exception as e:
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    logger.error(f"Image tokenization failed: {str(e)}")
    TOKENIZATION_ERRORS.labels(modality="image", error_type=type(e).__name__).inc()
    raise TokenizationError(f"Failed to tokenize image: {str(e)}", {"error": str(e)})

class AudioProcessor(ModalityProcessor):
 """Processor for audio inputs."""
 
 def __init__(self, config: TokenizerConfig):
  """
  Initialize audio processor with codec model.
  
  Args:
   config: Tokenizer configuration
  """
  super().__init__(config)
  self.codebooks = config.audio_codebooks
  self.codebook_size = config.audio_codebook_size
  self.sample_rate = config.audio_sample_rate
  
  # In production, we'd load actual models here
  try:
   model_path = config.model_paths["audio"]
   if not os.path.exists(model_path):
    logger.warning(f"Audio model not found at {model_path}. Using mock implementation.")
  except Exception as e:
   logger.error(f"Failed to load audio model: {str(e)}")

 async def preprocess(self, audio: torch.Tensor) -> torch.Tensor:
  """
  Preprocess audio input.
  
  Args:
   audio: Raw audio tensor of shape [C, T] or [T]
   
  Returns:
   Preprocessed audio tensor
   
  Raises:
   PreprocessingError: When audio preprocessing fails
   InvalidInputError: When input is not a valid audio tensor
  """
  if not isinstance(audio, torch.Tensor):
   raise InvalidInputError("Audio input must be a torch.Tensor", {"type": type(audio).__name__})
  
  try:
   # Ensure audio has shape [C, T]
   if len(audio.shape) == 1:
    audio = audio.unsqueeze(0)  # Add channel dimension
   elif len(audio.shape) > 2:
    raise InvalidInputError(
     "Audio must have shape [T] or [C, T]",
     {"shape": list(audio.shape)}
    )
   
   # Resample if necessary (simple implementation - in production use a proper resampling algorithm)
   # We're just implementing a mock version for the example
   return audio
  except Exception as e:
   logger.error(f"Audio preprocessing error: {str(e)}")
   raise PreprocessingError(f"Failed to preprocess audio: {str(e)}", {"error": str(e)})
 
 def tokenize(self, audio: torch.Tensor) -> TokenizationResult:
  """
  Tokenize preprocessed audio.
  
  Args:
   audio: Preprocessed audio tensor
   
  Returns:
   TokenizationResult with audio tokens and metadata
   
  Raises:
   TokenizationError: When tokenization fails
  """
  with self._timed_operation("Audio tokenization"):
   start_time = time.perf_counter()
   
   try:
    # Extract audio properties
    channels, samples = audio.shape
    
    # In production, this would use a real codec model
    # Here we'll implement a frame-based approach
    
    # Create frames (25ms per frame with 10ms stride is common)
    frame_length = int(0.025 * self.sample_rate)  # 25ms
    frame_stride = int(0.010 * self.sample_rate)  # 10ms
    num_frames = max(1, (samples - frame_length) // frame_stride + 1)
    
    # Extract features deterministically based on audio content
    # In reality this would be the output of a neural codec like EnCodec
    frames = []
    for i in range(min(num_frames, 1000)):  # Cap at 1000 frames
     start = i * frame_stride
     end = start + frame_length
     if end <= samples:
      frame = audio[:, start:end].mean(dim=0)
      frames.append(frame)
    
    if not frames:
     # Handle empty or very short audio
     frames = [torch.zeros(1)]
     num_frames = 1
    
    # Create "codebook" entries across multiple codebooks (RVQ-style)
    tokens = []
    for codebook in range(self.codebooks):
     # Generate deterministic tokens based on audio content
     codebook_tokens = []
     for i, frame in enumerate(frames):
      # Use frame statistics to generate token indices
      token_idx = (int(frame.abs().sum().item() * 1000) + i * codebook) % self.codebook_size
      codebook_tokens.append(token_idx)
     tokens.append(torch.tensor(codebook_tokens, dtype=torch.long))
    
    # Stack codebooks along a new dimension
    tokens = torch.stack(tokens, dim=1).unsqueeze(0)  # [1, num_frames, codebooks]
    
    # Create attention mask
    attn_mask = torch.ones(1, tokens.shape[1], dtype=torch.long)
    
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    TOKENIZATION_LATENCY.labels(modality="audio").observe(processing_time_ms / 1000)
    TOKENS_PROCESSED.labels(modality="audio").inc(tokens.numel())
    
    return TokenizationResult(
     tokens=tokens,
     attention_mask=attn_mask,
     modality=ModalityType.AUDIO,
     source_shape=audio.shape,
     processing_time_ms=processing_time_ms,
     cached=False,
     source_flag=torch.zeros(1, tokens.shape[1], dtype=torch.long),
     metadata={
      "type": "audio",
      "channels": channels,
      "sample_rate": self.sample_rate,
      "duration_sec": samples / self.sample_rate,
      "num_frames": num_frames,
      "codebooks": self.codebooks
     }
    )
   except Exception as e:
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    logger.error(f"Audio tokenization failed: {str(e)}")
    TOKENIZATION_ERRORS.labels(modality="audio", error_type=type(e).__name__).inc()
    raise TokenizationError(f"Failed to tokenize audio: {str(e)}", {"error": str(e)})

class VideoProcessor(ModalityProcessor):
 """Processor for video inputs."""
 
 def __init__(self, config: TokenizerConfig):
  """
  Initialize video processor with frame-based model.
  
  Args:
   config: Tokenizer configuration
  """
  super().__init__(config)
  self.frame_rate = config.video_frame_rate
  self.image_processor = ImageProcessor(config)

 async def preprocess(self, video: torch.Tensor) -> List[torch.Tensor]:
  """
  Preprocess video input.
  
  Args:
   video: Raw video tensor of shape [T, C, H, W]
   
  Returns:
   List of preprocessed frame tensors
   
  Raises:
   PreprocessingError: When video preprocessing fails
   InvalidInputError: When input is not a valid video tensor
  """
  if not isinstance(video, torch.Tensor):
   raise InvalidInputError("Video input must be a torch.Tensor", {"type": type(video).__name__})
  
  if len(video.shape) != 4:
   raise InvalidInputError(
    "Video must have shape [T, C, H, W]", 
    {"shape": list(video.shape)}
   )
  
  try:
   # Extract frames at desired frame rate
   num_frames = video.shape[0]
   selected_frames = []
   
   # Select frames based on target frame rate
   frame_interval = max(1, num_frames // (self.frame_rate * 5))  # Assumes 5-second video chunks
   
   tasks = []
   for i in range(0, num_frames, frame_interval):
    if i < num_frames:
     tasks.append(self.image_processor.preprocess(video[i]))
   
   # Process frames in parallel
   if tasks:
    selected_frames = await asyncio.gather(*tasks)
   
   if not selected_frames:
    raise PreprocessingError("No frames were extracted from video")
    
   return selected_frames
  except Exception as e:
   logger.error(f"Video preprocessing error: {str(e)}")
   raise PreprocessingError(f"Failed to preprocess video: {str(e)}", {"error": str(e)})
 
 def tokenize(self, frames: List[torch.Tensor]) -> TokenizationResult:
  """
  Tokenize preprocessed video frames.
  
  Args:
   frames: List of preprocessed frame tensors
   
  Returns:
   TokenizationResult with video tokens and metadata
   
  Raises:
   TokenizationError: When tokenization fails
  """
  with self._timed_operation("Video tokenization"):
   start_time = time.perf_counter()
   
   try:
    # Tokenize each frame using the image processor
    frame_results = []
    for frame in frames:
     result = self.image_processor.tokenize(frame)
     frame_results.append(result.tokens)
    
    if not frame_results:
     raise TokenizationError("No frames to tokenize in video")
    
    # Concatenate frame tokens with frame position markers
    tokens = torch.cat(frame_results, dim=1)
    
    # Create attention mask
    attn_mask = torch.ones_like(tokens)
    
    # Create frame position indicators (source flags)
    source_flags = torch.zeros_like(tokens)
    current_pos = 0
    for i, frame_tokens in enumerate(frame_results):
     frame_len = frame_tokens.size(1)
     source_flags[:, current_pos:current_pos+frame_len] = i
     current_pos += frame_len
    
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    TOKENIZATION_LATENCY.labels(modality="video").observe(processing_time_ms / 1000)
    TOKENS_PROCESSED.labels(modality="video").inc(tokens.numel())
    
    return TokenizationResult(
     tokens=tokens,
     attention_mask=attn_mask,
     modality=ModalityType.VIDEO,
     source_shape=(len(frames), *frames[0].shape),
     processing_time_ms=processing_time_ms,
     cached=False,
     source_flag=source_flags,
     metadata={
      "type": "video",
      "frame_count": len(frames),
      "frame_rate": self.frame_rate,
      "duration_sec": len(frames) / self.frame_rate,
      "frame_resolution": (frames[0].shape[1], frames[0].shape[2]) if frames else (0, 0)
     }
    )
   except Exception as e:
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    logger.error(f"Video tokenization failed: {str(e)}")
    TOKENIZATION_ERRORS.labels(modality="video", error_type=type(e).__name__).inc()
    raise TokenizationError(f"Failed to tokenize video: {str(e)}", {"error": str(e)})

class MultimodalTokenizer:
    """
    Production-grade tokenizer multiplexer for GPT-4o.
    Handles multi-modal inputs with async preprocessing, buffering, streaming, and source flags.
    Returns token embeddings + metadata for downstream fusion.
    
    Example:
        # Initialize tokenizer
        config = TokenizerConfig(
            max_text_length=2048,
            num_threads=8,
            enable_caching=True
        )
        tokenizer = MultimodalTokenizer(config)
        
        # Process multimodal inputs
        inputs = {
            "text": "Describe this image and audio",
            "image": image_tensor,
            "audio": audio_tensor
        }
        
        # Get tokenization results
        results = await tokenizer(inputs)
        
        # Access tokens by modality
        text_tokens = results["text"].tokens
        image_tokens = results["image"].tokens
        audio_tokens = results["audio"].tokens
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        """
        Initialize the multimodal tokenizer.
        
        Args:
            config: Tokenizer configuration. If None, uses default configuration.
        """
        self.config = config or TokenizerConfig()
        
        # Initialize processors for each modality
        self.processors = {
            ModalityType.TEXT: TextProcessor(self.config),
            ModalityType.IMAGE: ImageProcessor(self.config),
            ModalityType.AUDIO: AudioProcessor(self.config),
            ModalityType.VIDEO: VideoProcessor(self.config)
        }
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_threads)
        
        # Pre-tokenization buffer
        self._pre_tokenization_buffer = {m: [] for m in ModalityType}
        
        # Metrics
        self._metrics = {
            "total_processed": 0,
            "cache_hits": 0,
            "latency_ms": 0.0,
            "errors": 0,
            "tokens_per_modality": {m.value: 0 for m in ModalityType}
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized MultimodalTokenizer with config: {self.config.to_dict()}")
    
    async def _preprocess_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess inputs asynchronously for each modality.
        
        Args:
            inputs: Dictionary mapping modality names to inputs
            
        Returns:
            Dictionary of preprocessed inputs
            
        Raises:
            PreprocessingError: When preprocessing fails
            TimeoutError: When preprocessing times out
        """
        tasks = {}
        for modality_name, input_data in inputs.items():
            try:
                modality = ModalityType(modality_name)
                processor = self.processors[modality]
                tasks[modality_name] = processor.preprocess(input_data)
            except ValueError:
                logger.warning(f"Unknown modality: {modality_name}")
            except Exception as e:
                logger.error(f"Error setting up preprocessing for {modality_name}: {str(e)}")
                raise PreprocessingError(
                    f"Failed to preprocess {modality_name} input", 
                    {"error": str(e), "modality": modality_name}
                )
        
        if not tasks:
            return inputs
        
        # Set a timeout for preprocessing - moved outside try block
        timeout_sec = self.config.timeout_ms / 1000
        
        try:
            preprocessed_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Check for exceptions
            preprocessed = {}
            for (modality_name, _), result in zip(tasks.items(), preprocessed_results):
                if isinstance(result, Exception):
                    logger.error(f"Preprocessing error for {modality_name}: {str(result)}")
                    raise PreprocessingError(
                        f"Failed to preprocess {modality_name} input",
                        {"error": str(result), "modality": modality_name}
                    )
                preprocessed[modality_name] = result
            
            return preprocessed
        except asyncio.TimeoutError:
            logger.error(f"Preprocessing timed out after {timeout_sec}s")
            raise TimeoutError(
                f"Preprocessing timed out after {timeout_sec}s",
                {"timeout_ms": self.config.timeout_ms}
            )
    
    def _parallel_tokenize(self, preprocessed: Dict[str, Any]) -> List[TokenizationResult]:
        """
        Tokenize preprocessed inputs in parallel.
        
        Args:
            preprocessed: Dictionary of preprocessed inputs by modality
            
        Returns:
            List of TokenizationResult objects
            
        Raises:
            TokenizationError: When tokenization fails
        """
        futures: Dict[str, Future[TokenizationResult]] = {}
        
        for modality_name, input_data in preprocessed.items():
            try:
                modality = ModalityType(modality_name)
                processor = self.processors[modality]
                futures[modality_name] = self.thread_pool.submit(processor.tokenize, input_data)
            except ValueError:
                logger.warning(f"Unknown modality: {modality_name}")
            except Exception as e:
                logger.error(f"Error setting up tokenization for {modality_name}: {str(e)}")
                raise TokenizationError(
                    f"Failed to set up tokenization for {modality_name}", 
                    {"error": str(e), "modality": modality_name}
                )
        
        return self.retrieve_tokenization_results(futures)

    def retrieve_tokenization_results(self, futures):
        # Set a timeout for tokenization - moved outside try block
        timeout_sec = self.config.timeout_ms / 1000
        
        results = []
        for modality_name, future in futures.items():
            try:
                result = future.result(timeout=timeout_sec)
                results.append(result)
            except concurrent.futures.TimeoutError:
                logger.error(f"Tokenization timed out for {modality_name} after {timeout_sec}s")
                raise TimeoutError(
                    f"Tokenization timed out for {modality_name}",
                    {"timeout_ms": self.config.timeout_ms, "modality": modality_name}
                )
            except Exception as e:
                logger.error(f"Tokenization error for {modality_name}: {str(e)}")
                raise TokenizationError(
                    f"Failed to tokenize {modality_name} input", 
                    {"error": str(e), "modality": modality_name}
                )
        
        return results
    
    def _update_metrics(self, results: List[TokenizationResult]) -> None:
        """
        Update internal metrics based on tokenization results.
        
        Args:
            results: List of TokenizationResult objects
        """
        with self._lock:
            self._metrics["total_processed"] += len(results)
            
            for result in results:
                modality = result.modality.value
                if result.cached:
                    self._metrics["cache_hits"] += 1
                
                self._metrics["latency_ms"] += result.processing_time_ms
                self._metrics["tokens_per_modality"][modality] += result.tokens.numel()
    
    async def __call__(self, inputs: Dict[str, Any]) -> Dict[str, TokenizationResult]:
        """
        Process multimodal inputs and return tokenization results.
        
        Args:
            inputs: Dictionary mapping modality names to inputs
            
        Returns:
            Dictionary mapping modality names to TokenizationResult objects
            
        Raises:
            TokenizerError: When tokenization fails
        """
        if not inputs:
            return {}
            
        logger.info(f"Processing inputs with modalities: {list(inputs.keys())}")
        
        try:
            # Preprocess inputs
            preprocessed = await self._preprocess_input(inputs)
            
            # Tokenize in parallel
            results = self._parallel_tokenize(preprocessed)
            
            # Update metrics
            self._update_metrics(results)
            
            # Return results mapped by modality
            return {result.modality.value: result for result in results}
            
        except TokenizerError:
            # Re-raise tokenizer errors
            raise
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}", exc_info=True)
            self._metrics["errors"] += 1
            raise TokenizerError(
                f"Failed to process inputs: {str(e)}", 
                TokenizerErrorCode.UNKNOWN_ERROR,
                {"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def buffer_input(self, modality: ModalityType, data: Any) -> None:
        """
        Add input to pre-tokenization buffer for batch processing.
        
        Args:
            modality: Input modality
            data: Input data
        """
        with self._lock:
            self._pre_tokenization_buffer[modality].append(data)
            buffer_size = len(self._pre_tokenization_buffer[modality])
            QUEUE_DEPTH.labels(modality=modality.value).set(buffer_size)
            
            if buffer_size >= self.config.pre_tokenization_buffer:
                logger.info(f"Buffer full for {modality.value}, triggering tokenization.")
    
    def flush_buffer(self, modality: ModalityType) -> Optional[TokenizationResult]:
        """
        Process and clear the buffer for a specific modality.
        
        Args:
            modality: Modality to flush
            
        Returns:
            TokenizationResult if buffer had data, None otherwise
            
        Raises:
            TokenizationError: When tokenization fails
        """
        with self._lock:
            if not self._pre_tokenization_buffer[modality]:
                return None
                
            data = self._pre_tokenization_buffer[modality]
            self._pre_tokenization_buffer[modality] = []
            QUEUE_DEPTH.labels(modality=modality.value).set(0)
            
        try:
            # Process each item in the buffer individually
            results = [self.processors[modality].tokenize(item) for item in data]
            return results
        except Exception as e:
            logger.error(f"Buffer flush error for {modality.value}: {str(e)}")
            raise TokenizationError(
                f"Failed to process buffered {modality.value} input", 
                {"error": str(e), "modality": modality.value}
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current tokenizer metrics.
        
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            return {**self._metrics}
    
    def __del__(self) -> None:
        """Clean up resources when object is deleted."""
        try:
            self.thread_pool.shutdown(wait=True)
            logger.info("MultimodalTokenizer resources cleaned up")
        except Exception as e:
            logger.error(f"Error shutting down MultimodalTokenizer: {str(e)}")
