"""
Advanced Multimodal Tokenizer Adapter for GPT-Ø
Provides a unified interface to the MultimodalTokenizer with full modality support.

This adapter bridges the gap between the comprehensive modality system in gpt_model.py
and the tokenizer_mux.py multimodal tokenizer, ensuring all 13 modalities are supported.

This is the primary tokenization interface for GPT-Ø's self-modifying architecture,
handling both core modalities (TEXT, IMAGE, AUDIO, etc.) through tokenizer_mux.py 
and extended modalities (LIVE_WEB, LIDAR, GPS, CLOCK, RM_RF, ADS_B) through 
specialized fallback processors that align with gpt_model.py encoders.

Author: Morpheus
Date: August 3, 2025
Version: 2.0.0 - Full GPT-Ø Integration
"""

import json
import asyncio
import logging
import time
import hashlib
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum

import torch
import torch.nn.functional as F
import numpy as np

# Fix the import - remove the relative import issue  
try:
    from .tokenizer_mux import (
        MultimodalTokenizer, 
        TokenizerConfig, 
        TokenizationResult,
        ModalityType as TokenizerModalityType,
        TokenizerError,
        PreprocessingError,
        TokenizationError
    )
except ImportError:
    # Fallback for direct execution
    from tokenizer_mux import (
        MultimodalTokenizer, 
        TokenizerConfig, 
        TokenizationResult,
        ModalityType as TokenizerModalityType,
        TokenizerError,
        PreprocessingError,
        TokenizationError
    )

from tokenizers import SentencePieceBPETokenizer

# Configure logging
logger = logging.getLogger(__name__)

# Full modality support from gpt_model.py (13 modalities)
class ModalityType(Enum):
    """Complete modality support aligned with gpt_model.py (15 modalities total)"""
    TEXT = "text"
    STRUCTURED = "structured"  # For code, JSON, YAML, etc.
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TOOL = "tool"
    EMBEDDING = "embedding"
    LIVE_WEB = "live_web"      # Real-time web content processing
    LIDAR = "lidar"            # 3D spatial point cloud data
    GPS = "gps"                # Geographic coordinate systems
    CLOCK = "clock"            # Temporal/chronological data streams
    RM_RF = "rm_rf"            # File system operations (with safety mechanisms)
    ADS_B = "ads_b"            # Aircraft tracking and flight data
    EYES = "eyes"              # ISR (Intelligence, Surveillance, Reconnaissance)
    EARS = "ears"              # Spatial Domain Processing
    
    # NEW: Model Assimilation Modalities
    GGUF_MODEL = "gguf_model"
    ONNX_MODEL = "onnx_model"
    PYTORCH_MODEL = "pytorch_model"
    HUGGINGFACE_MODEL = "huggingface_model"
    RAW_BINARY = "raw_binary"
    NEURAL_PATTERNS = "neural_patterns"

class TokenizerAdapter:
    """
    Advanced multimodal tokenizer adapter that provides a clean, unified interface
    to the MultimodalTokenizer with support for all 13 GPT-Ø modalities.
    
    This adapter handles:
    - Full modality coverage (13 types vs 6 in base tokenizer_mux)
    - Async/sync interface bridging
    - Special token management
    - Fallback processing for unsupported modalities
    - Configuration management
    - Alignment with gpt_model.py encoders
    - Tool output head integration preparation
    """

    def __init__(self, config_path: Path):
        """
        Initializes the TokenizerAdapter with full modality support.

        Args:
            config_path: Path to the agent_config.json file.
        """
        self.config_path = config_path
        self._load_config()
        
        # Initialize the core multimodal tokenizer (handles 6 core modalities)
        tokenizer_config = TokenizerConfig(
            max_text_length=self.config.get("max_text_length", 4096),
            max_image_pixels=tuple(self.config.get("max_image_pixels", [768, 768])),
            audio_sample_rate=self.config.get("audio_sample_rate", 24000),
            enable_caching=self.config.get("enable_caching", True),
            cache_size=self.config.get("cache_size", 10000),
            num_threads=self.config.get("num_threads", 4),
            batch_size=self.config.get("batch_size", 32),
            timeout_ms=self.config.get("timeout_ms", 5000),
            circuit_breaker_max_failures=self.config.get("circuit_breaker_max_failures", 3),
            circuit_breaker_reset_sec=self.config.get("circuit_breaker_reset_sec", 30.0),
            pre_tokenization_buffer=self.config.get("pre_tokenization_buffer", 100),
            vqvae_codebook_size=self.config.get("vqvae_codebook_size", 8192),
            vqvae_embedding_dim=self.config.get("vqvae_embedding_dim", 256),
            audio_codebooks=self.config.get("audio_codebooks", 4),
            audio_codebook_size=self.config.get("audio_codebook_size", 1024),
            video_frame_rate=self.config.get("video_frame_rate", 30),
            model_paths=self.config.get("model_paths", {
                "text": "models/text_tokenizer.model",
                "image": "models/vqvae.pt", 
                "audio": "models/encodec.pt"
            })
        )
        
        self.multimodal_tokenizer = MultimodalTokenizer(tokenizer_config)
        
        # Initialize base text tokenizer for direct encode/decode operations
        self._init_base_text_tokenizer()
        
        # Special tokens for all 13 modalities
        self._init_special_tokens()
        
        # Model dimension for fallback processing alignment
        self.d_model = self.config.get("d_model", 4096)
        
        # Extended modality processors configuration
        self._init_extended_modality_configs()
        
        logger.info(f"TokenizerAdapter initialized with {len(ModalityType)} modalities")
        logger.info(f"Core modalities (tokenizer_mux): {[m.value for m in TokenizerModalityType]}")
        logger.info(f"Extended modalities (fallback): {[m.value for m in ModalityType if m.value not in [tm.value for tm in TokenizerModalityType]]}")
        logger.info(f"Model assimilation modalities: {[m.value for m in ModalityType if 'model' in m.value or m.value in ['raw_binary', 'neural_patterns']]}")

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            self.config = {}

    def _init_base_text_tokenizer(self) -> None:
        """Initialize base text tokenizer for direct operations."""
        try:
            model_path = self.config.get("tokenizer_model_path")
            if model_path and Path(model_path).exists():
                self.base_tokenizer = SentencePieceBPETokenizer(model_path)
            else:
                self.base_tokenizer = SentencePieceBPETokenizer()
                logger.warning("Using default SentencePieceBPE tokenizer")
        except Exception as e:
            logger.error(f"Failed to initialize base tokenizer: {e}")
            self.base_tokenizer = SentencePieceBPETokenizer()

    def _init_special_tokens(self) -> None:
        """Initialize special tokens for all modalities including model assimilation."""
        self.special_tokens = {
            # Core modality tokens (supported by tokenizer_mux)
            "<|text_start|>": 50257,
            "<|text_end|>": 50258,
            "<|structured_start|>": 50259,
            "<|structured_end|>": 50260,
            "<|image_start|>": 50261,
            "<|image_end|>": 50262,
            "<|audio_start|>": 50263,
            "<|audio_end|>": 50264,
            "<|tool_start|>": 50267,
            "<|tool_end|>": 50268,
            "<|embedding_start|>": 50269,
            "<|embedding_end|>": 50270,
            
            # Extended modality tokens for GPT-Ø (fallback processing)
            "<|live_web_start|>": 50271,
            "<|live_web_end|>": 50272,
            "<|lidar_start|>": 50273,
            "<|lidar_end|>": 50274,
            "<|gps_start|>": 50275,
            "<|gps_end|>": 50276,
            "<|clock_start|>": 50277,
            "<|clock_end|>": 50278,
            "<|rm_rf_start|>": 50279,
            "<|rm_rf_end|>": 50280,
            "<|ads_b_start|>": 50281,
            "<|ads_b_end|>": 50282,
            "<|video_start|>": 50265,
            "<|video_end|>": 50266,
            
            "<|eyes_start|>": 50303,
            "<|eyes_end|>": 50304,
            "<|ears_start|>": 50305,
            "<|ears_end|>": 50306,

            # NEW: Model Assimilation Tokens
            "<|gguf_model_start|>": 50307,
            "<|gguf_model_end|>": 50308,
            "<|onnx_model_start|>": 50309,
            "<|onnx_model_end|>": 50310,
            "<|pytorch_model_start|>": 50311,
            "<|pytorch_model_end|>": 50312,
            "<|huggingface_model_start|>": 50313,
            "<|huggingface_model_end|>": 50314,
            "<|raw_binary_start|>": 50315,
            "<|raw_binary_end|>": 50316,
            "<|neural_patterns_start|>": 50317,
            "<|neural_patterns_end|>": 50318,
            
            # Model Assimilation Control Tokens
            "<|assimilation_start|>": 50319,
            "<|assimilation_end|>": 50320,
            "<|capability_transfer|>": 50321,
            "<|weight_merge|>": 50322,
            "<|constitutional_check|>": 50323,
            "<|bayesian_select|>": 50324,

            # Reasoning and control tokens
            "<|reasoning_start|>": 50283,
            "<|reasoning_end|>": 50284,
            "<|cot_start|>": 50285,
            "<|cot_end|>": 50286,
            "<|self_modify|>": 50287,
            "<|tool_synthesis|>": 50288,
            "<|neural_exec|>": 50289,
            
            # Memory and context tokens
            "<|memory_start|>": 50290,
            "<|memory_end|>": 50291,
            "<|hot_memory|>": 50292,
            "<|cold_memory|>": 50293,
            "<|context_chunk|>": 50294,
            
            # Safety and validation tokens
            "<|safety_check|>": 50295,
            "<|validated|>": 50296,
            "<|error_state|>": 50297,
            "<|fallback|>": 50298,
            
            # Model state tokens
            "<|breathing_inhale|>": 50299,
            "<|breathing_exhale|>": 50300,
            "<|weight_update|>": 50301,
            "<|attention_focus|>": 50302,
        }
        
        # Create reverse mapping for token ID to string
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        logger.info(f"Initialized {len(self.special_tokens)} special tokens for model assimilation modalities")

    def _init_extended_modality_configs(self) -> None:
        """Initialize configuration for extended modalities that require fallback processing."""
        self.extended_modality_configs = {
            ModalityType.LIVE_WEB: {
                "max_content_length": self.config.get("live_web_max_content", 8192),
                "timeout_sec": self.config.get("live_web_timeout", 10.0),
                "encoding_method": "hash_based"
            },
            ModalityType.LIDAR: {
                "max_points": self.config.get("lidar_max_points", 65536),
                "spatial_dims": 3,
                "encoding_method": "point_cloud"
            },
            ModalityType.GPS: {
                "coordinate_precision": self.config.get("gps_precision", 6),
                "include_altitude": True,
                "encoding_method": "coordinate_tensor"
            },
            ModalityType.CLOCK: {
                "temporal_resolution": self.config.get("clock_resolution", "millisecond"),
                "max_sequence_length": self.config.get("clock_max_seq", 1024),
                "encoding_method": "temporal_series"
            },
            ModalityType.RM_RF: {
                "safety_validation": True,
                "allowed_operations": ["ls", "stat", "find", "locate"],
                "blocked_operations": ["rm", "rmdir", "unlink", "delete"],
                "encoding_method": "operation_tensor"
            },
            ModalityType.ADS_B: {
                "flight_data_fields": ["altitude", "latitude", "longitude", "speed", "heading", "callsign"],
                "encoding_method": "flight_vector"
            },
            ModalityType.VIDEO: {
                "max_frames": self.config.get("video_max_frames", 100),
                "frame_sampling": self.config.get("video_frame_sampling", "uniform"),
                "encoding_method": "frame_sequence"
            },
            ModalityType.EYES: {
                "max_detections": self.config.get("eyes_max_detections", 1024),
                "encoding_method": "structured_report"
            },
            ModalityType.EARS: {
                "max_signals": self.config.get("ears_max_signals", 2048),
                "encoding_method": "spatial_grid"
            },
            # NEW: Model Assimilation Modalities
            ModalityType.GGUF_MODEL: {
                "max_file_size": self.config.get("gguf_max_file_size", 10 * 1024 * 1024 * 1024),  # 10GB
                "safety_validation": True,
                "encoding_method": "model_assimilation"
            },
            ModalityType.ONNX_MODEL: {
                "max_file_size": self.config.get("onnx_max_file_size", 5 * 1024 * 1024 * 1024),  # 5GB
                "safety_validation": True,
                "encoding_method": "model_assimilation"
            },
            ModalityType.PYTORCH_MODEL: {
                "max_file_size": self.config.get("pytorch_max_file_size", 5 * 1024 * 1024 * 1024),  # 5GB
                "safety_validation": True,
                "encoding_method": "model_assimilation"
            },
            ModalityType.HUGGINGFACE_MODEL: {
                "max_file_size": self.config.get("hf_max_file_size", 5 * 1024 * 1024 * 1024),  # 5GB
                "safety_validation": True,
                "encoding_method": "model_assimilation"
            },
            ModalityType.RAW_BINARY: {
                "max_file_size": self.config.get("raw_max_file_size", 1 * 1024 * 1024 * 1024),  # 1GB
                "safety_validation": True,
                "encoding_method": "binary_assimilation"
            },
            ModalityType.NEURAL_PATTERNS: {
                "max_pattern_size": self.config.get("pattern_max_size", 1024 * 1024),  # 1MB
                "encoding_method": "pattern_recognition"
            }
        }
        
        # Reverse mapping for decoding
        self.token_to_id = self.special_tokens
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Base vocabulary size (GPT-2 base is 50257)
        self._vocab_size = max(self.special_tokens.values()) + 1

    def encode(self, text: str) -> List[int]:
        """
        Encodes text into a sequence of token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        try:
            # Handle special tokens first
            for token, token_id in self.special_tokens.items():
                if token in text:
                    # For now, simple replacement - could be enhanced
                    text = text.replace(token, f" {token} ")
            
            # Use base tokenizer for encoding
            encoding = self.base_tokenizer.encode(text)
            token_ids = encoding.ids
            
            # Replace special token text with actual IDs
            final_ids = []
            i = 0
            while i < len(token_ids):
                # Check if current position starts a special token
                found_special = False
                for token, token_id in self.special_tokens.items():
                    # Simple approach - in production would need more sophisticated matching
                    pass
                
                if not found_special:
                    final_ids.append(token_ids[i])
                i += 1
            
            return final_ids
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return []

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a sequence of token IDs back into text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        try:
            # Replace special token IDs with placeholder text first
            processed_ids = []
            for token_id in token_ids:
                if token_id in self.id_to_token:
                    # Handle special tokens - convert back to text representation
                    special_token = self.id_to_token[token_id]
                    # For now, we'll include the special token in output
                    # In production, might want to handle differently
                    processed_ids.append(token_id)  # Keep as-is for now
                else:
                    processed_ids.append(token_id)
            
            # Use base tokenizer for decoding
            decoded = self.base_tokenizer.decode(processed_ids)
            return decoded
            
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            return ""

    @property
    def vocab_size(self) -> int:
        """Returns the total size of the vocabulary including special tokens."""
        return self._vocab_size

    def token_id(self, token: str) -> Optional[int]:
        """
        Gets the ID of a token, including special tokens.
        
        Args:
            token: Token string to look up
            
        Returns:
            Token ID if found, None otherwise
        """
        # Check special tokens first
        if token in self.special_tokens:
            return self.special_tokens[token]
        
        # Check base tokenizer
        try:
            encoding = self.base_tokenizer.encode(token)
            if encoding.ids:
                return encoding.ids[0]
        except Exception as e:
            logger.debug(f"Token lookup failed for '{token}': {e}")
        
        return None

    async def encode_multimodal(self, inputs: Dict[str, Any]) -> Dict[str, TokenizationResult]:
        """
        Encode multimodal inputs using the full tokenizer pipeline.
        
        Args:
            inputs: Dictionary mapping modality names to input data
            
        Returns:
            Dictionary mapping modality names to TokenizationResult objects
        """
        # Map GPT-Ø modality types to tokenizer types where supported
        mapped_inputs = {}
        unsupported_modalities = []
        
        for modality_name, data in inputs.items():
            try:
                gpt_modality = ModalityType(modality_name)
                
                # Map to tokenizer modalities where possible
                if gpt_modality in [ModalityType.TEXT, ModalityType.STRUCTURED, 
                                  ModalityType.IMAGE, ModalityType.AUDIO, 
                                  ModalityType.TOOL, ModalityType.EMBEDDING]:
                    mapped_inputs[modality_name] = data
                else:
                    # Handle extended modalities with fallback processing
                    unsupported_modalities.append((gpt_modality, data))
                    
            except ValueError:
                logger.warning(f"Unknown modality type: {modality_name}")
        
        # Process supported modalities through multimodal tokenizer
        results = {}
        if mapped_inputs:
            try:
                tokenizer_results = await self.multimodal_tokenizer(mapped_inputs)
                results.update(tokenizer_results)
            except Exception as e:
                logger.error(f"Multimodal tokenization failed: {e}")
        
        # Handle unsupported modalities with fallback processing
        for modality, data in unsupported_modalities:
            try:
                fallback_result = self._process_extended_modality(modality, data)
                results[modality.value] = fallback_result
            except Exception as e:
                logger.error(f"Failed to process {modality.value}: {e}")
        
        return results

    def _process_extended_modality(self, modality: ModalityType, data: Any) -> TokenizationResult:
        """
        Process extended modalities not supported by base tokenizer.
        
        Args:
            modality: The extended modality type
            data: Input data for the modality
            
        Returns:
            TokenizationResult with processed tokens
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        try:
            if modality in [ModalityType.GGUF_MODEL, ModalityType.ONNX_MODEL, 
                           ModalityType.PYTORCH_MODEL, ModalityType.HUGGINGFACE_MODEL,
                           ModalityType.RAW_BINARY, ModalityType.NEURAL_PATTERNS]:
                # Route to GGUF assimilator for model processing
                tokens = self._process_model_assimilation(data, modality)
            elif modality == ModalityType.LIVE_WEB:
                tokens = self._process_live_web_data(data)
            elif modality == ModalityType.LIDAR:
                tokens = self._process_lidar_data(data)
            elif modality == ModalityType.GPS:
                tokens = self._process_gps_data(data)
            elif modality == ModalityType.CLOCK:
                tokens = self._process_temporal_data(data)
            elif modality == ModalityType.RM_RF:
                tokens = self._process_file_operation_data(data)
            elif modality == ModalityType.ADS_B:
                tokens = self._process_ads_b_data(data)
            elif modality == ModalityType.VIDEO:
                tokens = self._process_video_data(data)
            elif modality == ModalityType.EYES:
                tokens = self._process_eyes_data(data)
            elif modality == ModalityType.EARS:
                tokens = self._process_ears_data(data)
            else:
                # Generic fallback
                tokens = self._generic_fallback_tokenization(data)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time_ms = start_time.elapsed_time(end_time)
            else:
                processing_time_ms = 0.0
            
            attention_mask = torch.ones_like(tokens)
            
            return TokenizationResult(
                tokens=tokens,
                attention_mask=attention_mask,
                modality=TokenizerModalityType.EMBEDDING,  # Use embedding as fallback type
                source_shape=tokens.shape,
                processing_time_ms=processing_time_ms,
                cached=False
            )
            
        except Exception as e:
            logger.error(f"Extended modality processing failed for {modality.value}: {e}")
            # Return empty result on failure
            empty_tokens = torch.zeros((1, 1), dtype=torch.long)
            return TokenizationResult(
                tokens=empty_tokens,
                attention_mask=torch.ones_like(empty_tokens),
                modality=TokenizerModalityType.EMBEDDING,
                source_shape=empty_tokens.shape,
                processing_time_ms=0.0,
                cached=False
            )

    def _process_model_assimilation(self, inputs: Any, modality: ModalityType) -> torch.Tensor:
        """Process model assimilation inputs through GGUF assimilator."""
        
        # Import here to avoid circular dependencies
        try:
            from extra_output_heads.gguf_assimilator_modality_encoder import GGUFAssimilatorModalityEncoder
        except ImportError:
            logger.warning("GGUF assimilator not available, using fallback tokenization")
            return self._generic_fallback_tokenization(inputs)
        
        # Create assimilator instance (should be cached)
        if not hasattr(self, '_gguf_assimilator'):
            self._gguf_assimilator = GGUFAssimilatorModalityEncoder(
                input_dim=self.d_model,
                hidden_dim=self.d_model * 2,
                output_dim=self.d_model
            )
        
        try:
            # Convert model path/data to tokenized representation
            if isinstance(inputs, str) and os.path.exists(inputs):
                # File path input
                model_type = modality.value.replace('_model', '')
                assimilation_result = self._gguf_assimilator.assimilate_model(inputs, model_type)
                
                if assimilation_result and assimilation_result.success:
                    # Convert assimilated representation to tokens
                    token_tensor = self._convert_assimilation_to_tokens(assimilation_result)
                    return token_tensor
                else:
                    logger.warning(f"Model assimilation failed for {inputs}")
                    return self._create_fallback_result(inputs, modality)
            else:
                # Direct tensor/data input
                return self._process_direct_model_data(inputs, modality)
                
        except Exception as e:
            logger.error(f"Model assimilation processing failed: {e}")
            return self._create_fallback_result(inputs, modality)

    def _convert_assimilation_to_tokens(self, assimilation_result) -> torch.Tensor:
        """Convert assimilation result to token representation."""
        # This creates a tokenized representation of the assimilated model
        # that can be processed by the main transformer
        
        # Start with capability embedding
        capability_tokens = []
        for capability in assimilation_result.assimilated_capabilities:
            # Convert capability names to token IDs
            cap_hash = hash(capability) % 50000  # Simple hash to token ID
            capability_tokens.append(cap_hash)
        
        # Add performance metrics as tokens
        performance_tokens = []
        for metric, value in assimilation_result.performance_gain.items():
            # Quantize performance values to token range
            quantized_value = int(value * 1000) % 50000
            performance_tokens.append(quantized_value)
        
        # Combine into final token sequence
        all_tokens = capability_tokens + performance_tokens
        
        # Pad or truncate to expected sequence length
        target_length = min(512, len(all_tokens))  # Reasonable sequence length
        if len(all_tokens) < target_length:
            all_tokens.extend([0] * (target_length - len(all_tokens)))  # Pad with 0
        else:
            all_tokens = all_tokens[:target_length]  # Truncate
        
        return torch.tensor(all_tokens, dtype=torch.long).unsqueeze(0)

    def _process_direct_model_data(self, inputs: Any, modality: ModalityType) -> torch.Tensor:
        """Process direct model data (tensors, arrays, etc.)."""
        if isinstance(inputs, torch.Tensor):
            # For tensor inputs, apply appropriate modality tokens
            start_token_key = f"<|{modality.value}_start|>"
            end_token_key = f"<|{modality.value}_end|>"
            
            start_token = self.special_tokens.get(start_token_key, self.special_tokens["<|neural_patterns_start|>"])
            end_token = self.special_tokens.get(end_token_key, self.special_tokens["<|neural_patterns_end|>"])
            
            # Flatten and quantize tensor for tokenization
            flattened = inputs.flatten()[:1024]  # Limit size
            quantized = (torch.abs(flattened) * 1000).long()  # Simple quantization
            
            # Combine with special tokens
            start_tensor = torch.tensor([start_token], dtype=torch.long)
            end_tensor = torch.tensor([end_token], dtype=torch.long)
            
            return torch.cat([start_tensor, quantized, end_tensor]).unsqueeze(0)
        else:
            # Fallback to generic processing
            return self._generic_fallback_tokenization(inputs)

    def _create_fallback_result(self, inputs: Any, modality: ModalityType) -> torch.Tensor:
        """Create fallback result for failed model assimilation."""
        try:
            # Try to convert to string and tokenize
            data_str = str(inputs)
            start_token_key = f"<|{modality.value}_start|>"
            end_token_key = f"<|{modality.value}_end|>"
            
            # Add modality-specific tokens if available
            if start_token_key in self.special_tokens and end_token_key in self.special_tokens:
                data_str = f"{start_token_key}{data_str}{end_token_key}"
            
            encoded = self.encode(data_str)
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        except:
            return torch.zeros((1, 1), dtype=torch.long)

    def _process_live_web_data(self, data: Any) -> torch.Tensor:
        """Process live web data into tokens."""
        if isinstance(data, str):
            # URL or web content
            encoded = self.encode(f"<|live_web_start|>{data}<|live_web_end|>")
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        elif isinstance(data, dict):
            # Structured web data
            web_text = json.dumps(data)
            encoded = self.encode(f"<|live_web_start|>{web_text}<|live_web_end|>")
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        else:
            return torch.zeros((1, 1), dtype=torch.long)

    def _process_lidar_data(self, data: Any) -> torch.Tensor:
        """Process LiDAR point cloud data into tokens."""
        if isinstance(data, (np.ndarray, torch.Tensor)):
            # Convert point cloud to tensor representation
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            
            # Flatten and quantize point cloud for tokenization
            flattened = data.flatten()[:1024]  # Limit size
            quantized = (flattened * 1000).long()  # Simple quantization
            
            # Add special tokens
            start_token = torch.tensor([self.special_tokens["<|lidar_start|>"]], dtype=torch.long)
            end_token = torch.tensor([self.special_tokens["<|lidar_end|>"]], dtype=torch.long)
            
            return torch.cat([start_token, quantized, end_token]).unsqueeze(0)
        else:
            return torch.zeros((1, 1), dtype=torch.long)

    def _process_eyes_data(self, data: Any) -> torch.Tensor:
        """Process ISR (eyes) data into tokens."""
        if isinstance(data, dict):
            # Structured ISR report
            report_text = json.dumps(data)
            encoded = self.encode(f"<|eyes_start|>{report_text}<|eyes_end|>")
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        else:
            return torch.zeros((1, 1), dtype=torch.long)

    def _process_ears_data(self, data: Any) -> torch.Tensor:
        """Process spatial domain (ears) data into tokens."""
        if isinstance(data, dict):
            # Structured spatial intelligence report
            report_text = json.dumps(data)
            encoded = self.encode(f"<|ears_start|>{report_text}<|ears_end|>")
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        else:
            return torch.zeros((1, 1), dtype=torch.long)

    def _process_gps_data(self, data: Any) -> torch.Tensor:
        """Process GPS coordinate data into tokens."""
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            # [lat, lon, alt?] format
            lat, lon = data[0], data[1]
            alt = data[2] if len(data) > 2 else 0.0
            
            # Quantize coordinates
            lat_q = int((lat + 90) * 10000)  # Normalize and quantize latitude
            lon_q = int((lon + 180) * 10000)  # Normalize and quantize longitude
            alt_q = int(alt * 100)  # Quantize altitude
            
            tokens = torch.tensor([
                self.special_tokens["<|gps_start|>"],
                lat_q % 50257,  # Keep within vocab range
                lon_q % 50257,
                alt_q % 50257,
                self.special_tokens["<|gps_end|>"]
            ], dtype=torch.long)
            
            return tokens.unsqueeze(0)
        else:
            return torch.zeros((1, 1), dtype=torch.long)

    def _process_temporal_data(self, data: Any) -> torch.Tensor:
        """Process temporal/clock data into tokens."""
        if isinstance(data, (int, float)):
            # Unix timestamp
            timestamp = int(data)
        elif isinstance(data, str):
            # Try to parse timestamp from string
            try:
                import time
                timestamp = int(time.mktime(time.strptime(data, "%Y-%m-%d %H:%M:%S")))
            except:
                timestamp = 0
        else:
            timestamp = 0
        
        # Encode timestamp
        tokens = torch.tensor([
            self.special_tokens["<|clock_start|>"],
            timestamp % 50257,  # Keep within vocab range
            self.special_tokens["<|clock_end|>"]
        ], dtype=torch.long)
        
        return tokens.unsqueeze(0)

    def _process_file_operation_data(self, data: Any) -> torch.Tensor:
        """Process file removal/deletion operation data into tokens."""
        if isinstance(data, str):
            # File path or operation description
            encoded = self.encode(f"<|rm_rf_start|>{data}<|rm_rf_end|>")
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        elif isinstance(data, dict):
            # Operation metadata
            op_text = json.dumps(data)
            encoded = self.encode(f"<|rm_rf_start|>{op_text}<|rm_rf_end|>")
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        else:
            return torch.zeros((1, 1), dtype=torch.long)

    def _process_ads_b_data(self, data: Any) -> torch.Tensor:
        """Process ADS-B aircraft tracking data into tokens."""
        if isinstance(data, dict):
            # Aircraft data: {icao, callsign, lat, lon, alt, speed, heading, etc.}
            ads_b_text = json.dumps(data)
            encoded = self.encode(f"<|ads_b_start|>{ads_b_text}<|ads_b_end|>")
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        elif isinstance(data, str):
            # Raw ADS-B message
            encoded = self.encode(f"<|ads_b_start|>{data}<|ads_b_end|>")
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        else:
            return torch.zeros((1, 1), dtype=torch.long)

    def _process_video_data(self, data: Any) -> torch.Tensor:
        """Process video data into tokens (fallback for when not supported by base tokenizer)."""
        if isinstance(data, torch.Tensor):
            # Video tensor [T, C, H, W]
            # Simple approach: flatten and quantize
            flattened = data.flatten()[:512]  # Limit size
            quantized = (flattened * 1000).long()
            
            start_token = torch.tensor([self.special_tokens["<|video_start|>"]], dtype=torch.long)
            end_token = torch.tensor([self.special_tokens["<|video_end|>"]], dtype=torch.long)
            
            return torch.cat([start_token, quantized, end_token]).unsqueeze(0)
        else:
            return torch.zeros((1, 1), dtype=torch.long)

    def _generic_fallback_tokenization(self, data: Any) -> torch.Tensor:
        """Generic fallback tokenization for unknown data types."""
        try:
            # Try to convert to string and tokenize
            data_str = str(data)
            encoded = self.encode(data_str)
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        except:
            return torch.zeros((1, 1), dtype=torch.long)

    def encode_modality(self, data: Any, modality: ModalityType) -> torch.Tensor:
        """
        Encode data for a specific modality (sync interface).
        
        Args:
            data: Input data
            modality: Modality type
            
        Returns:
            Encoded tensor
        """
        try:
            # Run async method in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.encode_multimodal({modality.value: data})
                )
                if modality.value in result:
                    return result[modality.value].tokens
                else:
                    return torch.zeros((1, 1), dtype=torch.long)
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Modality encoding failed for {modality.value}: {e}")
            return torch.zeros((1, 1), dtype=torch.long)

    def get_modality_vocab_size(self, modality: ModalityType) -> int:
        """Get vocabulary size for a specific modality."""
        # For now, return same size for all modalities
        # In production, different modalities might have different vocab sizes
        return self.vocab_size

    def get_special_token_ids(self) -> Dict[str, int]:
        """Get mapping of special tokens to their IDs."""
        return self.special_tokens.copy()

    def supports_modality(self, modality: ModalityType) -> bool:
        """Check if a modality is supported."""
        return modality in ModalityType  # All modalities are supported via fallback

    def get_metrics(self) -> Dict[str, Any]:
        """Get tokenizer metrics."""
        try:
            return self.multimodal_tokenizer.get_metrics()
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    def get_token_representation(self, text: str, modalities: Optional[List[str]] = None) -> torch.Tensor:
        """
        Get token representation for input text and modalities.
        
        This is the main method used by the GPT-Ø model to tokenize inputs
        for processing through the neural network.
        
        Args:
            text: Input text to tokenize
            modalities: List of modality types to consider (optional)
            
        Returns:
            torch.Tensor: Token IDs as tensor for model input
        """
        try:
            # Default to text modality if none specified
            if modalities is None:
                modalities = ['text']
            
            # Primary text tokenization
            if 'text' in modalities or not modalities:
                token_ids = self.encode(text)
                return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension
            
            # Handle multimodal inputs
            inputs = {'text': text}
            
            # For now, we primarily handle text but prepare for multimodal expansion
            # This method will be enhanced as multimodal capabilities are integrated
            for modality in modalities:
                if modality != 'text':
                    # For non-text modalities, add special tokens to indicate modality context
                    if f"<{modality.upper()}>" in self.special_tokens:
                        # Prepend modality token to text
                        modality_token_id = self.special_tokens[f"<{modality.upper()}>"]
                        text_tokens = self.encode(text)
                        combined_tokens = [modality_token_id] + text_tokens
                        return torch.tensor(combined_tokens, dtype=torch.long).unsqueeze(0)
            
            # Fallback to text tokenization
            token_ids = self.encode(text)
            return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Token representation failed: {e}")
            # Return minimal valid token sequence
            return torch.tensor([[self.special_tokens.get('<UNK>', 0)]], dtype=torch.long)
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Tensor of token IDs to decode
            
        Returns:
            str: Decoded text
        """
        try:
            # Convert tensor to list of integers
            if isinstance(token_ids, torch.Tensor):
                if token_ids.dim() > 1:
                    # Remove batch dimensions
                    token_ids = token_ids.squeeze()
                token_list = token_ids.tolist()
            else:
                token_list = token_ids
            
            # Use the base text tokenizer to decode
            decoded_text = self.decode(token_list)
            return decoded_text
            
        except Exception as e:
            logger.error(f"Token decoding failed: {e}")
            return "[DECODE_ERROR]"


if __name__ == '__main__':
    # Example usage
    config_file = Path(__file__).parent.parent / "config" / "agent_config.json"
    adapter = TokenizerAdapter(config_path=config_file)

    print(f"Tokenizer vocabulary size: {adapter.vocab_size}")

    image_start_id = adapter.token_id("<|image_start|>")
    print(f"Image start token ID: {image_start_id}")

    text = "Hello, world!"
    encoded = adapter.encode(text)
    decoded = adapter.decode(encoded)

    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    assert text == decoded
    print("\n✅ Tokenizer adapter works as expected.")


if __name__ == '__main__':
    # Example usage
    config_file = Path(__file__).parent.parent / "config" / "agent_config.json"
    adapter = TokenizerAdapter(config_path=config_file)

    print(f"Tokenizer vocabulary size: {adapter.vocab_size}")

    image_start_id = adapter.token_id("<|image_start|>")
    print(f"Image start token ID: {image_start_id}")

    text = "Hello, world!"
    encoded = adapter.encode(text)
    decoded = adapter.decode(encoded)

    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    assert text == decoded
    print("\n✅ Tokenizer adapter works as expected.")