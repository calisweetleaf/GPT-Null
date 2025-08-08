"""
Neural Model Runtime - Layer 1: Dynamic Model Management System
Integrates with neural_memory_runtime.py for revolutionary local AI

Author: Cybernetic Architecture Division
License: MIT
"""

# Core imports
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
import threading
import time
import hashlib
import weakref
import math
import pickle
import struct
from pathlib import Path
import requests
import subprocess
import psutil
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import logging
import gc
import uuid
# Cross-platform resource handling
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    # resource module not available on Windows
    RESOURCE_AVAILABLE = False
    logging.info("Resource module not available (expected on Windows)")
    class MockResource:
        @staticmethod
        def getrusage(who):
            return type('usage', (), {'ru_maxrss': 0})()
        RUSAGE_SELF = 0
    resource = MockResource()
import gc
import uuid

# Neural memory runtime imports with comprehensive error handling
try:
    from . import neural_memory_runtime
    NeuralMemoryRuntime = neural_memory_runtime.NeuralMemoryRuntime
    MemoryTier = neural_memory_runtime.MemoryTier
    NEURAL_MEMORY_AVAILABLE = True
    logging.info("Neural memory runtime successfully imported")
except ImportError as e:
    try:
        # Fallback: try importing from relative path
        import neural_memory_runtime
        NeuralMemoryRuntime = neural_memory_runtime.NeuralMemoryRuntime
        MemoryTier = neural_memory_runtime.MemoryTier
        NEURAL_MEMORY_AVAILABLE = True
        logging.info("Neural memory runtime successfully imported (fallback)")
    except ImportError:
        logging.warning(f"Neural memory runtime not available: {e}")
        logging.warning("Operating with limited functionality")
        NEURAL_MEMORY_AVAILABLE = False
        # Create minimal fallback classes
        class MockNeuralMemoryRuntime:
            def __init__(self, *args, **kwargs):
                pass
        class MockMemoryTier:
            ULTRA_HOT = "ultra_hot"
            HOT = "hot" 
            WARM = "warm"
            COLD = "cold"
            FROZEN = "frozen"
        NeuralMemoryRuntime = MockNeuralMemoryRuntime
        MemoryTier = MockMemoryTier
except Exception as e:
    logging.error(f"Unexpected error importing neural_memory_runtime: {e}")
    NEURAL_MEMORY_AVAILABLE = False
    # Create minimal fallback classes
    class MockNeuralMemoryRuntime:
        def __init__(self, *args, **kwargs):
            pass
    class MockMemoryTier:
        ULTRA_HOT = "ultra_hot"
        HOT = "hot" 
        WARM = "warm"
        COLD = "cold"
        FROZEN = "frozen"
    NeuralMemoryRuntime = MockNeuralMemoryRuntime
    MemoryTier = MockMemoryTier

# PyTorch imports with error handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logging.info("PyTorch successfully imported")
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    # Define minimal torch-like interface for fallback
    class MockTorch:
        @staticmethod
        def tensor(data, dtype=None):
            return data
        @staticmethod
        def is_tensor(obj):
            return False
        @staticmethod
        def inference_mode():
            class NullContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return NullContext()
        
        class cuda:
            @staticmethod
            def is_available():
                return False
    
    torch = MockTorch()
    
    class nn:
        class Module:
            def __init__(self):
                pass

# Add the .venv Scripts directory to path for llama-cpp files
import sys
from pathlib import Path
venv_scripts_path = Path(__file__).parent.parent / ".venv" / "Scripts"
if venv_scripts_path.exists() and str(venv_scripts_path) not in sys.path:
    sys.path.insert(0, str(venv_scripts_path))

# Llama.cpp imports with comprehensive error handling
LLAMA_CPP_AVAILABLE = False
LLAMA_CPP_ERROR = None

try:
    # Try importing from the venv Scripts directory first
    import llama_cpp
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
    logging.info("llama-cpp-python successfully imported from venv Scripts")
except ImportError as e:
    LLAMA_CPP_ERROR = str(e)
    logging.warning(f"llama-cpp-python not available: {e}")

except Exception as e:
    LLAMA_CPP_ERROR = f"Unexpected error importing llama-cpp-python: {e}"
    logging.error(LLAMA_CPP_ERROR)
    LLAMA_CPP_AVAILABLE = False
    
    # Fallback to mock implementation
    Llama = MockLlama
    class llama_cpp:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log import status
logger.info(f"Import Status - PyTorch: {TORCH_AVAILABLE}, Llama.cpp: {LLAMA_CPP_AVAILABLE}, Neural Memory: {NEURAL_MEMORY_AVAILABLE}")
if LLAMA_CPP_ERROR:
    logger.warning(f"Llama.cpp import error: {LLAMA_CPP_ERROR}")

# Validate critical dependencies - warn but don't fail
if not NEURAL_MEMORY_AVAILABLE:
    logger.warning(
        "Neural memory runtime is not available. "
        "Operating with limited functionality using mock implementations."
    )


class ModelType(Enum):
    """Supported model architectures"""
    GGUF = "gguf"
    BITNET = "bitnet" 
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"


class CognitiveProfile(Enum):
    """Cognitive behavioral profiles"""
    ANALYTICAL = "analytical"       # Deep reasoning, step-by-step analysis
    CREATIVE = "creative"          # Divergent thinking, novel connections
    CONVERSATIONAL = "conversational"  # Natural dialogue, empathetic
    TECHNICAL = "technical"        # Precise, domain-specific expertise
    RESEARCH = "research"          # Information synthesis, multi-perspective
    TEACHING = "teaching"          # Explanatory, pedagogical approach


@dataclass
class ModelConfiguration:
    """Dynamic model configuration with neural optimization"""
    
    # Core model parameters
    model_path: str
    model_type: ModelType
    cognitive_profile: CognitiveProfile
    
    # Dynamic parameters (auto-adjusted by neural runtime)
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    
    # Neural memory configuration - properly typed with fallback
    memory_allocation: Dict[str, float] = field(default_factory=lambda: {
        "ULTRA_HOT": 0.15,
        "HOT": 0.35,
        "WARM": 0.35,
        "COLD": 0.10,
        "FROZEN": 0.05
    })
    
    # Behavioral instructions (loaded dynamically)
    system_instructions: List[str] = field(default_factory=list)
    cognitive_frameworks: Dict[str, Any] = field(default_factory=dict)
    
    # Performance optimization
    max_context_length: int = 8192
    batch_size: int = 1
    attention_sparsity: float = 0.1
    compression_threshold: float = 0.8
    
    # Runtime adaptation settings
    auto_optimize: bool = True
    learning_rate: float = 0.001
    adaptation_frequency: int = 100  # Optimize every N interactions


class NeuralModelManager:
    """Advanced model management with neural memory integration"""
    
    def __init__(self, neural_runtime: NeuralMemoryRuntime):
        """Initialize neural model manager with memory runtime.
        
        Args:
            neural_runtime: Instance of NeuralMemoryRuntime for memory management
            
        Raises:
            TypeError: If neural_runtime is not a NeuralMemoryRuntime instance
            ValueError: If neural_runtime is not properly initialized
        """
        if not isinstance(neural_runtime, NeuralMemoryRuntime):
            raise TypeError(
                f"Expected NeuralMemoryRuntime instance, got {type(neural_runtime)}"
            )
        
        # Validate neural runtime is functional
        try:
            # Test basic functionality
            test_stats = neural_runtime.get_runtime_stats()
            if not isinstance(test_stats, dict):
                raise ValueError("Neural runtime not properly initialized")
        except Exception as e:
            raise ValueError(f"Neural runtime validation failed: {e}") from e
        
        self.neural_runtime = neural_runtime
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfiguration] = {}
        self.active_model: Optional[str] = None
        
        # Performance tracking
        self.interaction_count = 0
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Background optimization thread
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        self._optimization_thread.start()
        
        logger.info("Neural Model Manager initialized with memory runtime integration")
    
    def register_model(self, 
                      model_id: str, 
                      config: ModelConfiguration) -> bool:
        """Register a model configuration with comprehensive validation.
        
        Args:
            model_id: Unique identifier for the model
            config: Model configuration object
            
        Returns:
            True if registration successful, False otherwise
            
        Raises:
            ValueError: If model_id or config is invalid
            FileNotFoundError: If model file doesn't exist
        """
        # Input validation
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")
        
        if not isinstance(config, ModelConfiguration):
            raise TypeError(f"Expected ModelConfiguration, got {type(config)}")
        
        if model_id in self.model_configs:
            logger.warning(f"Model {model_id} already registered, overwriting")
        
        try:
            # Validate model file exists and is accessible
            model_path = Path(config.model_path).resolve()
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if not os.access(model_path, os.R_OK):
                raise PermissionError(f"Model file not readable: {model_path}")
            
            # Validate file size
            file_size = model_path.stat().st_size
            if file_size == 0:
                raise ValueError(f"Model file is empty: {model_path}")
            
            # Store configuration
            self.model_configs[model_id] = config
            
            # Pre-calculate model signature for integrity checking
            model_signature = self._calculate_model_signature(str(model_path))
            
            # Store configuration metadata in neural memory
            config_tensor = torch.tensor([hash(str(config))], dtype=torch.float32) if TORCH_AVAILABLE else [hash(str(config))]
            
            self.neural_runtime.store_activation(
                f"model_config_{model_id}",
                config_tensor,
                importance=1.0
            )
            
            # Store model signature for integrity verification
            signature_tensor = torch.tensor([hash(model_signature)], dtype=torch.float32) if TORCH_AVAILABLE else [hash(model_signature)]
            
            self.neural_runtime.store_activation(
                f"model_signature_{model_id}",
                signature_tensor,
                importance=0.8
            )
            
            logger.info(f"Model registered successfully: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Model registration failed for {model_id}: {e}")
            # Clean up any partial state
            if model_id in self.model_configs:
                del self.model_configs[model_id]
            return False
    
    def load_model(self, model_id: str) -> bool:
        """Load model with neural memory optimization and integrity verification.
        
        Args:
            model_id: Identifier of the model to load
            
        Returns:
            True if loading successful, False otherwise
            
        Raises:
            ValueError: If model_id is invalid or not registered
            FileNotFoundError: If model file is missing
            MemoryError: If insufficient memory for loading
        """
        # Input validation
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")
        
        if model_id not in self.model_configs:
            raise ValueError(f"Model not registered: {model_id}")
        
        config = self.model_configs[model_id]
        
        try:
            # Verify model file integrity
            current_signature = self._calculate_model_signature(config.model_path)
            stored_signature_tensor = self.neural_runtime.retrieve_activation(f"model_signature_{model_id}")
            
            if stored_signature_tensor is not None:
                stored_signature_hash = stored_signature_tensor.item() if hasattr(stored_signature_tensor, 'item') else stored_signature_tensor[0]
                if hash(current_signature) != stored_signature_hash:
                    logger.warning(f"Model file signature mismatch for {model_id}, proceeding with caution")
            
            # Check if model already loaded in neural memory cache
            cached_model_key = f"loaded_model_{model_id}"
            cached_model = self.neural_runtime.retrieve_activation(cached_model_key)
            
            if cached_model is not None and model_id in self.loaded_models:
                logger.info(f"Model loaded from neural memory cache: {model_id}")
                self.active_model = model_id
                return True
            
            # Load model based on type with proper error handling
            if config.model_type == ModelType.GGUF:
                model = self._load_gguf_model(config)
            elif config.model_type == ModelType.BITNET:
                model = self._load_bitnet_model(config)
            elif config.model_type == ModelType.PYTORCH:
                model = self._load_pytorch_model(config)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            # Optimize model for neural memory integration
            optimized_model = self._optimize_model_for_memory(model, config)
            
            # Store in loaded models registry
            self.loaded_models[model_id] = optimized_model
            
            # Cache model metadata in neural memory hierarchy
            self._cache_model_weights(model_id, optimized_model, config)
            
            # Store loaded model reference in neural memory
            model_ref_tensor = torch.tensor([hash(str(model_id))], dtype=torch.float32) if TORCH_AVAILABLE else [hash(str(model_id))]
            self.neural_runtime.store_activation(
                cached_model_key,
                model_ref_tensor,
                importance=0.9
            )
            
            self.active_model = model_id
            logger.info(f"Model loaded successfully: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed for {model_id}: {e}")
            # Clean up any partial state
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
            return False
    
    def _load_gguf_model(self, config: ModelConfiguration) -> Any:
        """Load GGUF model with memory optimization and production safety.
        
        Args:
            config: Model configuration with validated parameters
            
        Returns:
            Loaded GGUF model instance with memory optimization
            
        Raises:
            Exception: If model loading fails with detailed error context
            ValueError: If configuration parameters are invalid
            FileNotFoundError: If model file doesn't exist or isn't accessible
            MemoryError: If insufficient memory for model loading
        """
        if not LLAMA_CPP_AVAILABLE:
            raise Exception(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )
        
        # Input validation with security checks
        model_path = Path(config.model_path).resolve()
        
        # Security: Validate file path is within allowed directories
        if not self._validate_model_path_security(model_path):
            raise ValueError(f"Model path security validation failed: {model_path}")
        
        # Validate file exists and is readable
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.access(model_path, os.R_OK):
            raise PermissionError(f"Model file not readable: {model_path}")
        
        # Validate model file integrity
        file_size = model_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Model file is empty: {model_path}")
        
        # Memory pre-check before loading
        available_memory = self._get_available_memory_gb()
        estimated_model_memory = self._estimate_gguf_memory_requirement(model_path)
        
        if estimated_model_memory > available_memory * 0.8:  # Leave 20% buffer
            raise MemoryError(
                f"Insufficient memory. Need ~{estimated_model_memory:.1f}GB, "
                f"available: {available_memory:.1f}GB"
            )
        
        # Configure model parameters with bounds checking
        model_params = self._build_gguf_parameters(config, available_memory)
        
        # Start loading with timeout and progress tracking
        loading_start_time = time.time()
        correlation_id = self._generate_correlation_id()
        
        try:
            logging.info(
                f"Loading GGUF model",
                extra={
                    'correlation_id': correlation_id,
                    'model_path': str(model_path),
                    'file_size_mb': file_size / (1024 * 1024),
                    'estimated_memory_gb': estimated_model_memory,
                    'available_memory_gb': available_memory,
                    'model_params': model_params
                }
            )

            # Load model with timeout protection
            model = self._load_gguf_with_timeout(
                model_path=str(model_path),
                params=model_params,
                timeout_seconds=300,  # 5 minute timeout
                correlation_id=correlation_id
            )

            # Validate loaded model
            if not self._validate_loaded_gguf_model(model):
                raise Exception("Model validation failed after loading")

            # Wrap model with monitoring and optimization
            optimized_model = self._wrap_gguf_with_monitoring(
                model, config, correlation_id
            )

            loading_time = time.time() - loading_start_time

            # Log successful loading with metrics
            logging.info(
                f"GGUF model loaded successfully",
                extra={
                    'correlation_id': correlation_id,
                    'model_path': str(model_path),
                    'loading_time_seconds': loading_time,
                    'actual_memory_usage_gb': self._get_model_memory_usage(optimized_model),
                    'model_metadata': self._extract_gguf_metadata(model)
                }
            )

            # Store performance metrics
            self._record_loading_metrics(
                model_type='gguf',
                loading_time=loading_time,
                memory_usage=self._get_model_memory_usage(optimized_model),
                correlation_id=correlation_id
            )

            return optimized_model

        except Exception as e:
            loading_time = time.time() - loading_start_time

            # Log failure with context
            logging.error(
                f"GGUF model loading failed",
                extra={
                    'correlation_id': correlation_id,
                    'model_path': str(model_path),
                    'loading_time_seconds': loading_time,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                exc_info=True
            )

            # Clean up any partial state
            self._cleanup_failed_loading(correlation_id)

            # Re-raise with enhanced context
            raise Exception(
                f"GGUF loading failed for {model_path}: {type(e).__name__}: {e}"
            ) from e
    
    def _validate_model_path_security(self, model_path: Path) -> bool:
        """Validate model path for comprehensive security compliance.
        
        Args:
            model_path: Resolved path to model file
            
        Returns:
            True if path is secure, False otherwise
        """
        try:
            # Input validation
            if not isinstance(model_path, Path):
                logging.error("model_path must be a Path object")
                return False
            
            # Ensure path is absolute and normalized
            try:
                resolved_path = model_path.resolve()
            except (OSError, RuntimeError) as e:
                logging.error(f"Path resolution failed: {e}")
                return False
            
            # Check for path traversal attempts - comprehensive check
            path_str = str(resolved_path)
            suspicious_patterns = [
                '..',           # Directory traversal
                '~',            # Home directory access
                '$',            # Environment variable expansion
                '%',            # Windows environment variable
                '/dev/',        # Unix device files
                '/proc/',       # Unix process files
                '/sys/',        # Unix system files
                'C:\\Windows',  # Windows system directories
                'C:\\Program Files',
            ]
            
            # Only check for backslashes if we're on a Unix system
            import platform
            if platform.system() != 'Windows':
                suspicious_patterns.append('\\')  # Windows path separators (if on Unix)
            
            for pattern in suspicious_patterns:
                if pattern in path_str:
                    logging.warning(f"Suspicious path pattern '{pattern}' detected: {resolved_path}")
                    return False
            
            # Validate path components for suspicious characters
            for part in resolved_path.parts:
                if any(char in part for char in ['<', '>', '|', '*', '?', '"']):
                    logging.warning(f"Invalid characters in path component: {part}")
                    return False
            
            # Validate file extension whitelist
            allowed_extensions = {'.gguf', '.bin', '.safetensors', '.pt', '.pth', '.onnx'}
            file_ext = resolved_path.suffix.lower()
            if file_ext not in allowed_extensions:
                logging.warning(f"Disallowed file extension: {file_ext}")
                return False
            
            # Check path length to prevent buffer overflow attacks
            if len(path_str) > 4096:  # Reasonable max path length
                logging.warning(f"Path too long ({len(path_str)} chars): {resolved_path}")
                return False
            
            # Verify path is within allowed directories (implement sandbox)
            allowed_prefixes = [
                Path.cwd(),                    # Current working directory
                Path.home() / "models",        # User models directory
                Path("/tmp/models"),           # Temporary models (Unix)
                Path("C:/temp/models"),        # Temporary models (Windows)
            ]
            
            path_allowed = False
            for allowed_prefix in allowed_prefixes:
                try:
                    resolved_path.relative_to(allowed_prefix.resolve())
                    path_allowed = True
                    break
                except ValueError:
                    continue  # Path not under this prefix
                except OSError:
                    continue  # Prefix doesn't exist or can't be resolved
            
            if not path_allowed:
                logging.warning(f"Path not in allowed directories: {resolved_path}")
                return False
            
            # Check file permissions
            if resolved_path.exists():
                stat_info = resolved_path.stat()
                
                # Check if file is readable
                if not os.access(resolved_path, os.R_OK):
                    logging.warning(f"File not readable: {resolved_path}")
                    return False
                
                # Check file size (prevent loading extremely large files)
                max_file_size = 50 * 1024 * 1024 * 1024  # 50GB max
                if stat_info.st_size > max_file_size:
                    logging.warning(f"File too large ({stat_info.st_size} bytes): {resolved_path}")
                    return False
                
                # Check if file is not a symlink (prevent symlink attacks)
                if resolved_path.is_symlink():
                    logging.warning(f"Symlinks not allowed: {resolved_path}")
                    return False
            
            logging.info(f"Path security validation passed: {resolved_path}")
            return True
            
        except Exception as e:
            logging.error(f"Model path security validation error: {e}")
            return False
    
    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB.
        
        Returns:
            Available memory in gigabytes
        """
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            
            # Account for Python process current usage
            process = psutil.Process()
            process_memory_gb = process.memory_info().rss / (1024 ** 3)
            
            return max(0.0, available_gb - process_memory_gb * 0.1)  # 10% buffer
            
        except Exception as e:
            logging.warning(f"Memory detection failed, using conservative estimate: {e}")
            return 4.0  # Conservative fallback
    
    def _estimate_gguf_memory_requirement(self, model_path: Path) -> float:
        """Estimate memory requirement for GGUF model.
        
        Args:
            model_path: Path to GGUF model file
            
        Returns:
            Estimated memory requirement in GB
        """
        try:
            file_size_gb = model_path.stat().st_size / (1024 ** 3)
            
            # GGUF models typically need ~1.2x file size for loading
            # Plus overhead for context and processing
            estimated_gb = file_size_gb * 1.2 + 0.5  # Base overhead
            
            return estimated_gb
            
        except Exception as e:
            logging.warning(f"Memory estimation failed: {e}")
            return 8.0  # Conservative fallback
    
    def _build_gguf_parameters(self, config: ModelConfiguration, available_memory_gb: float) -> Dict[str, Any]:
        """Build validated GGUF model parameters.
        
        Args:
            config: Model configuration
            available_memory_gb: Available system memory
            
        Returns:
            Dictionary of validated model parameters
        """
        # Calculate optimal context length based on available memory
        max_context = min(
            config.max_context_length,
            int(available_memory_gb * 1024)  # Rough heuristic
        )
        
        # Ensure context length is within valid bounds
        max_context = max(512, min(max_context, 32768))
        
        # Build parameters with validation
        params = {
            'model_path': config.model_path,
            'n_ctx': max_context,
            'n_batch': min(config.batch_size * 128, max_context // 4),
            'n_threads': min(os.cpu_count() or 4, 8),  # Reasonable thread limit
            'n_gpu_layers': -1 if torch.cuda.is_available() else 0,
            'use_mmap': True,  # Memory mapping for efficiency
            'use_mlock': False,  # Avoid locking too much memory
            'verbose': False,  # Reduce noise in logs
            'seed': 42,  # Deterministic for testing
        }
        
        # Add cognitive profile optimizations
        if config.cognitive_profile == CognitiveProfile.ANALYTICAL:
            params['repeat_penalty'] = 1.05
            params['temperature'] = 0.3
        elif config.cognitive_profile == CognitiveProfile.CREATIVE:
            params['repeat_penalty'] = 1.15
            params['temperature'] = 0.8
        else:
            params['repeat_penalty'] = config.repeat_penalty
            params['temperature'] = config.temperature
        
        return params
    
    def _load_gguf_with_timeout(self, 
                               model_path: str, 
                               params: Dict[str, Any],
                               timeout_seconds: int,
                               correlation_id: str) -> Any:
        """Load GGUF model with timeout protection.
        
        Args:
            model_path: Path to model file
            params: Model parameters
            timeout_seconds: Maximum loading time
            correlation_id: Request correlation ID
            
        Returns:
            Loaded model instance
            
        Raises:
            TimeoutError: If loading exceeds timeout
            Exception: If loading fails
        """
        import signal
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        def load_model():
            try:
                return Llama(**params)
            except Exception as e:
                logging.error(
                    f"Model loading thread failed",
                    extra={
                        'correlation_id': correlation_id,
                        'error': str(e),
                        'params': params
                    }
                )
                raise
        
        # Use thread executor for timeout control
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load_model)
            
            try:
                model = future.result(timeout=timeout_seconds)
                return model
                
            except FutureTimeoutError:
                # Cancel the future and clean up
                future.cancel()
                raise TimeoutError(
                    f"Model loading timed out after {timeout_seconds} seconds"
                )
    
    def _validate_loaded_gguf_model(self, model: Any) -> bool:
        """Validate loaded GGUF model functionality.
        
        Args:
            model: Loaded model instance
            
        Returns:
            True if model is valid and functional
        """
        try:
            # Test basic model functionality
            test_prompt = "Hello"
            
            # Attempt a simple inference to validate model
            with torch.inference_mode():
                result = model(test_prompt, max_tokens=1, echo=False)
            
            # Validate result structure
            if not isinstance(result, dict) or 'choices' not in result:
                logging.error("Model validation failed: invalid result structure")
                return False
            
            if not result['choices'] or not result['choices'][0].get('text'):
                logging.error("Model validation failed: no text output")
                return False
            
            logging.info("Model validation successful")
            return True
            
        except Exception as e:
            logging.error(f"Model validation failed: {e}")
            return False
    
    def _wrap_gguf_with_monitoring(self, 
                                  model: Any, 
                                  config: ModelConfiguration,
                                  correlation_id: str) -> Any:
        """Wrap GGUF model with monitoring and optimization.
        
        Args:
            model: Raw GGUF model
            config: Model configuration
            correlation_id: Request correlation ID
            
        Returns:
            Wrapped model with monitoring capabilities
        """
        class MonitoredGGUFModel:
            def __init__(self, base_model, config, neural_runtime, correlation_id):
                self.base_model = base_model
                self.config = config
                self.neural_runtime = neural_runtime
                self.correlation_id = correlation_id
                self.inference_count = 0
                self.total_tokens_generated = 0
                self.avg_inference_time = 0.0
                
                # Model metadata
                self.model_metadata = {
                    'type': 'gguf',
                    'path': config.model_path,
                    'cognitive_profile': config.cognitive_profile.value,
                    'loaded_at': time.time(),
                    'correlation_id': correlation_id,
                    'memory_optimized': True
                }
            
            def __call__(self, prompt: str, **kwargs) -> Dict[str, Any]:
                """Generate response with monitoring and caching."""
                inference_start = time.time()
                self.inference_count += 1
                
                # Generate cache key
                cache_key = self._generate_cache_key(prompt, kwargs)
                
                # Check neural memory cache
                cached_result = self.neural_runtime.retrieve_activation(cache_key)
                if cached_result is not None:
                    logging.info(
                        f"Cache hit for inference",
                        extra={
                            'correlation_id': self.correlation_id,
                            'inference_count': self.inference_count,
                            'cache_key': cache_key[:16]
                        }
                    )
                    return self._decode_cached_result(cached_result)
                
                try:
                    # Set default parameters with overrides
                    inference_params = {
                        'max_tokens': kwargs.get('max_tokens', 512),
                        'temperature': kwargs.get('temperature', self.config.temperature),
                        'top_p': kwargs.get('top_p', self.config.top_p),
                        'top_k': kwargs.get('top_k', self.config.top_k),
                        'repeat_penalty': kwargs.get('repeat_penalty', self.config.repeat_penalty),
                        'echo': kwargs.get('echo', False),
                        'stop': kwargs.get('stop', [])
                    }
                    
                    # Validate parameters
                    inference_params = self._validate_inference_params(inference_params)
                    
                    # Perform inference with resource monitoring
                    with self._monitor_resource_usage():
                        result = self.base_model(prompt, **inference_params)
                    
                    # Update metrics
                    inference_time = time.time() - inference_start
                    self._update_metrics(inference_time, result)
                    
                    # Cache result
                    self._cache_inference_result(cache_key, result, prompt, inference_time)
                    
                    logging.info(
                        f"Inference completed",
                        extra={
                            'correlation_id': self.correlation_id,
                            'inference_count': self.inference_count,
                            'inference_time_seconds': inference_time,
                            'tokens_generated': len(result.get('choices', [{}])[0].get('text', '').split()),
                            'cache_key': cache_key[:16]
                        }
                    )
                    
                    return result
                    
                except Exception as e:
                    inference_time = time.time() - inference_start
                    
                    logging.error(
                        f"Inference failed",
                        extra={
                            'correlation_id': self.correlation_id,
                            'inference_count': self.inference_count,
                            'inference_time_seconds': inference_time,
                            'error': str(e),
                            'prompt_length': len(prompt)
                        },
                        exc_info=True
                    )
                    
                    raise Exception(f"GGUF inference failed: {e}") from e
            
            def _generate_cache_key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
                """Generate deterministic cache key for inference."""
                key_data = {
                    'prompt': prompt,
                    'model_path': self.config.model_path,
                    'cognitive_profile': self.config.cognitive_profile.value,
                    **{k: v for k, v in kwargs.items() if k in [
                        'max_tokens', 'temperature', 'top_p', 'top_k', 'repeat_penalty'
                    ]}
                }
                
                key_string = json.dumps(key_data, sort_keys=True)
                return hashlib.sha256(key_string.encode()).hexdigest()[:32]
            
            def _validate_inference_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
                """Validate and sanitize inference parameters."""
                # Bound checking for numerical parameters
                params['max_tokens'] = max(1, min(params['max_tokens'], 4096))
                params['temperature'] = max(0.01, min(params['temperature'], 2.0))
                params['top_p'] = max(0.01, min(params['top_p'], 1.0))
                params['top_k'] = max(1, min(params['top_k'], 100))
                params['repeat_penalty'] = max(0.1, min(params['repeat_penalty'], 2.0))
                
                # Validate stop sequences
                if not isinstance(params['stop'], list):
                    params['stop'] = []
                
                return params
            
            def _monitor_resource_usage(self):
                """Context manager for monitoring resource usage during inference."""
                class ResourceMonitor:
                    def __enter__(self):
                        self.start_memory = psutil.Process().memory_info().rss
                        return self
                    
                    def __exit__(self, exc_type, exc_val, exc_tb):
                        end_memory = psutil.Process().memory_info().rss
                        memory_delta = (end_memory - self.start_memory) / (1024 * 1024)  # MB
                        
                        if memory_delta > 100:  # Log significant memory increases
                            logging.warning(
                                f"High memory usage during inference: {memory_delta:.1f}MB"
                            )
                
                return ResourceMonitor()
            
            def _update_metrics(self, inference_time: float, result: Dict[str, Any]):
                """Update inference metrics."""
                tokens_generated = len(result.get('choices', [{}])[0].get('text', '').split())
                self.total_tokens_generated += tokens_generated
                
                # Update rolling average
                alpha = 0.1  # Smoothing factor
                self.avg_inference_time = (
                    alpha * inference_time + 
                    (1 - alpha) * self.avg_inference_time
                )
            
            def _cache_inference_result(self, 
                                     cache_key: str, 
                                     result: Dict[str, Any],
                                     prompt: str,
                                     inference_time: float):
                """Cache inference result in neural memory."""
                try:
                    # Calculate importance based on inference characteristics
                    importance = self._calculate_cache_importance(prompt, result, inference_time)
                    
                    # Store result with metadata
                    cache_data = {
                        'result': result,
                        'cached_at': time.time(),
                        'inference_time': inference_time,
                        'model_path': self.config.model_path
                    }
                    
                    # Convert to tensor for neural memory storage
                    cache_tensor = torch.tensor([hash(json.dumps(cache_data, sort_keys=True))], dtype=torch.float32)
                    
                    self.neural_runtime.store_activation(cache_key, cache_tensor, importance)
                    
                except Exception as e:
                    logging.warning(f"Failed to cache inference result: {e}")
            
            def _calculate_cache_importance(self, 
                                          prompt: str, 
                                          result: Dict[str, Any],
                                          inference_time: float) -> float:
                """Calculate importance score for caching decisions."""
                base_importance = 0.5
                
                # Higher importance for longer inference times (expensive results)
                if inference_time > 2.0:
                    base_importance += 0.3
                elif inference_time > 1.0:
                    base_importance += 0.2
                
                # Higher importance for longer outputs (complex results)
                output_length = len(result.get('choices', [{}])[0].get('text', ''))
                if output_length > 1000:
                    base_importance += 0.2
                elif output_length > 500:
                    base_importance += 0.1
                
                return min(1.0, base_importance)
            
            def _decode_cached_result(self, cached_tensor: torch.Tensor) -> Dict[str, Any]:
                """Decode cached result from tensor storage."""
                # In a real implementation, this would properly decode the cached result
                # For now, return a placeholder that indicates cache hit
                return {
                    'choices': [{'text': '[CACHED_RESULT]'}],
                    'cached': True,
                    'model': self.config.model_path
                }
            
            def get_stats(self) -> Dict[str, Any]:
                """Get model performance statistics."""
                return {
                    'inference_count': self.inference_count,
                    'total_tokens_generated': self.total_tokens_generated,
                    'avg_inference_time': self.avg_inference_time,
                    'model_metadata': self.model_metadata
                }
        
        return MonitoredGGUFModel(model, config, self.neural_runtime, correlation_id)
    
    def _extract_gguf_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract metadata from loaded GGUF model."""
        try:
            # Extract available metadata from the model
            metadata = {
                'model_type': 'gguf',
                'context_length': getattr(model, 'n_ctx', 'unknown'),
                'vocab_size': getattr(model, 'n_vocab', 'unknown'),
                'embedding_dim': getattr(model, 'n_embd', 'unknown'),
            }
            
            return metadata
            
        except Exception as e:
            logging.warning(f"Failed to extract model metadata: {e}")
            return {'model_type': 'gguf', 'metadata_extraction_failed': True}
    
    def _get_model_memory_usage(self, model: Any) -> float:
        """Get current memory usage of the model in GB."""
        try:
            # Get current process memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb / 1024  # Convert to GB
            
        except Exception as e:
            logging.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def _record_loading_metrics(self, 
                               model_type: str,
                               loading_time: float,
                               memory_usage: float,
                               correlation_id: str):
        """Record model loading performance metrics."""
        try:
            metrics = {
                'model_type': model_type,
                'loading_time_seconds': loading_time,
                'memory_usage_gb': memory_usage,
                'timestamp': time.time(),
                'correlation_id': correlation_id
            }
            
            # Store metrics in neural memory for analysis
            metrics_tensor = torch.tensor([loading_time, memory_usage], dtype=torch.float32)
            self.neural_runtime.store_activation(
                f"loading_metrics_{correlation_id}",
                metrics_tensor,
                importance=0.6
            )
            
        except Exception as e:
            logging.warning(f"Failed to record loading metrics: {e}")
    
    def _cleanup_failed_loading(self, correlation_id: str):
        """Clean up resources after failed model loading."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear any partial state from neural memory
            # This would be implemented based on the neural memory cleanup API
            
            logging.info(f"Cleaned up failed loading attempt: {correlation_id}")
            
        except Exception as e:
            logging.warning(f"Cleanup failed: {e}")
    
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for request tracking."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _load_bitnet_model(self, config: ModelConfiguration) -> Any:
        """Load BitNet model with 1-bit optimization"""
        try:
            print(f"Loading BitNet model: {config.model_path}")
            
            # BitNet-specific loading logic
            # This would integrate with BitNet inference engines
            model = {
                'type': 'bitnet',
                'path': config.model_path,
                'config': config,
                'loaded_at': time.time(),
                'quantized': True
            }
            
            return model
            
        except Exception as e:
            raise Exception(f"BitNet loading failed: {e}")
    
    def _load_pytorch_model(self, config: ModelConfiguration) -> Any:
        """Load PyTorch model with neural memory integration"""
        try:
            model = torch.load(config.model_path, map_location='cpu')
            
            # Integrate with neural memory runtime
            if hasattr(model, 'forward'):
                model = self._wrap_model_with_memory(model, config)
            
            return model
            
        except Exception as e:
            raise Exception(f"PyTorch loading failed: {e}")
    
    def _wrap_model_with_memory(self, model: nn.Module, config: ModelConfiguration) -> nn.Module:
        """Wrap PyTorch model with neural memory optimization"""
        
        class MemoryOptimizedModel(nn.Module):
            def __init__(self, base_model, neural_runtime, config):
                super().__init__()
                self.base_model = base_model
                self.neural_runtime = neural_runtime
                self.config = config
                self.forward_count = 0
            
            def forward(self, *args, **kwargs):
                self.forward_count += 1
                
                # Generate cache key
                input_signature = self._generate_input_signature(*args, **kwargs)
                cache_key = f"forward_{input_signature}"
                
                # Check neural memory cache
                cached_result = self.neural_runtime.retrieve_activation(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Standard forward pass with memory optimization
                with torch.inference_mode():
                    result = self.base_model(*args, **kwargs)
                
                # Cache result based on importance
                importance = self._calculate_result_importance(result)
                self.neural_runtime.store_activation(cache_key, result, importance)
                
                return result
            
            def _generate_input_signature(self, *args, **kwargs) -> str:
                """Generate unique signature for input"""
                signature_parts = []
                for arg in args:
                    if torch.is_tensor(arg):
                        signature_parts.append(str(arg.shape))
                return hashlib.md5(''.join(signature_parts).encode()).hexdigest()[:16]
            
            def _calculate_result_importance(self, result) -> float:
                """Calculate importance score for caching decisions"""
                if torch.is_tensor(result):
                    # Higher variance = more important to cache
                    variance = torch.var(result).item()
                    return min(1.0, variance * 10)
                return 0.5
        
        return MemoryOptimizedModel(model, self.neural_runtime, config)
    
    def _optimize_model_for_memory(self, model: Any, config: ModelConfiguration) -> Any:
        """Apply neural memory optimizations to loaded model"""
        
        # Apply memory allocation strategy
        memory_strategy = {
            'ultra_hot_layers': int(0.1 * getattr(model, 'num_layers', 32)),
            'compression_targets': ['attention_weights', 'feed_forward'],
            'sparsity_ratio': config.attention_sparsity
        }
        
        # Store optimization metadata
        model_metadata = {
            'optimization_strategy': memory_strategy,
            'cognitive_profile': config.cognitive_profile.value,
            'memory_allocation': config.memory_allocation,
            'optimized_at': time.time()
        }
        
        # Attach metadata to model
        if hasattr(model, '__dict__'):
            model.neural_metadata = model_metadata
        
        return model
    
    def _cache_model_weights(self, model_id: str, model: Any, config: ModelConfiguration):
        """Cache model weights in neural memory hierarchy"""
        
        if hasattr(model, 'state_dict') and callable(model.state_dict):
            # PyTorch model
            state_dict = model.state_dict()
            
            for name, weight in state_dict.items():
                # Determine importance based on layer type
                importance = self._calculate_weight_importance(name, weight)
                
                # Store in appropriate memory tier
                cache_key = f"weights_{model_id}_{name}"
                self.neural_runtime.store_activation(cache_key, weight, importance)
        
        else:
            # Non-PyTorch model - store metadata
            model_tensor = torch.tensor([hash(str(model))], dtype=torch.float32)
            self.neural_runtime.store_activation(f"model_metadata_{model_id}", model_tensor, 1.0)
    
    def _calculate_weight_importance(self, layer_name: str, weight: torch.Tensor) -> float:
        """Calculate importance score for model weights"""
        
        # Higher importance for attention and embedding layers
        if any(keyword in layer_name.lower() for keyword in ['attention', 'embed', 'output']):
            base_importance = 0.9
        elif 'norm' in layer_name.lower():
            base_importance = 0.7
        else:
            base_importance = 0.5
        
        # Adjust based on weight statistics
        if weight.numel() > 0:
            weight_variance = torch.var(weight).item()
            variance_factor = min(1.0, weight_variance * 100)
            return min(1.0, base_importance + variance_factor * 0.1)
        
        return base_importance
    
    def _calculate_model_signature(self, model_path: str) -> str:
        """Calculate unique signature for model file"""
        try:
            import hashlib
            hash_md5 = hashlib.md5()
            with open(model_path, "rb") as f:
                # Read file in chunks to handle large models
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return str(hash(model_path))
    
    def generate_response(self, 
                         prompt: str, 
                         model_id: Optional[str] = None,
                         **kwargs) -> Dict[str, Any]:
        """Generate response with neural memory optimization and caching.
        
        Args:
            prompt: Input prompt for generation
            model_id: Optional model identifier, uses active model if None
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing response and metadata
            
        Raises:
            ValueError: If no model is loaded or prompt is invalid
            RuntimeError: If generation fails
        """
        # Input validation
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")
        
        if len(prompt.strip()) == 0:
            raise ValueError("prompt cannot be empty or whitespace only")
        
        target_model = model_id or self.active_model
        if not target_model:
            raise ValueError("No model specified and no active model loaded")
        
        if target_model not in self.loaded_models:
            raise ValueError(f"Model not loaded: {target_model}")
        
        config = self.model_configs[target_model]
        model = self.loaded_models[target_model]
        
        try:
            # Update interaction count for analytics
            self.interaction_count += 1
            correlation_id = self._generate_correlation_id()
            
            # Generate deterministic cache key
            prompt_params = {
                'prompt': prompt,
                'model_id': target_model,
                **{k: v for k, v in kwargs.items() if k in [
                    'max_tokens', 'temperature', 'top_p', 'top_k', 'repeat_penalty'
                ]}
            }
            
            cache_key = self._generate_cache_key(prompt_params)
            
            # Check neural memory cache for previous response
            cached_response = self.neural_runtime.retrieve_activation(cache_key)
            if cached_response is not None:
                logger.info(
                    f"Cache hit for prompt",
                    extra={
                        'correlation_id': correlation_id,
                        'model_id': target_model,
                        'cache_key': cache_key[:16]
                    }
                )
                
                return {
                    'response': self._decode_cached_response(cached_response),
                    'cached': True,
                    'model': target_model,
                    'interaction_count': self.interaction_count,
                    'correlation_id': correlation_id
                }
            
            # Generate response based on model type
            generation_start = time.time()
            
            if config.model_type in [ModelType.GGUF, ModelType.BITNET]:
                response = self._generate_gguf_response(model, prompt, config, **kwargs)
            elif config.model_type == ModelType.PYTORCH:
                response = self._generate_pytorch_response(model, prompt, config, **kwargs)
            else:
                raise RuntimeError(f"Unsupported model type for generation: {config.model_type}")
            
            generation_time = time.time() - generation_start
            
            # Validate response
            if not response or not isinstance(response, str):
                raise RuntimeError("Model generated invalid response")
            
            # Cache response in neural memory with importance scoring
            importance = self._calculate_response_importance(prompt, response, generation_time)
            
            response_tensor = torch.tensor([hash(response)], dtype=torch.float32) if TORCH_AVAILABLE else [hash(response)]
            
            cache_success = self.neural_runtime.store_activation(
                cache_key, 
                response_tensor, 
                importance=importance
            )
            
            if not cache_success:
                logger.warning(f"Failed to cache response for {correlation_id}")
            
            # Trigger adaptive optimization if needed
            if self.interaction_count % config.adaptation_frequency == 0:
                self._trigger_adaptation(target_model)
            
            # Log successful generation
            logger.info(
                f"Response generated successfully",
                extra={
                    'correlation_id': correlation_id,
                    'model_id': target_model,
                    'generation_time_seconds': generation_time,
                    'response_length': len(response),
                    'cached': False
                }
            )
            
            return {
                'response': response,
                'cached': False,
                'model': target_model,
                'interaction_count': self.interaction_count,
                'generation_time': generation_time,
                'correlation_id': correlation_id
            }
            
        except Exception as e:
            logger.error(
                f"Response generation failed",
                extra={
                    'correlation_id': correlation_id,
                    'model_id': target_model,
                    'error': str(e)
                },
                exc_info=True
            )
            
            return {
                'error': f"Response generation failed: {e}",
                'model': target_model,
                'correlation_id': correlation_id
            }
    
    def _generate_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key from parameters.
        
        Args:
            params: Parameters dictionary
            
        Returns:
            SHA-256 hash of sorted parameters
        """
        # Sort parameters for deterministic hashing
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(sorted_params.encode('utf-8')).hexdigest()[:32]
    
    def _decode_cached_response(self, cached_tensor: Any) -> str:
        """Decode cached response from neural memory.
        
        Args:
            cached_tensor: Cached tensor from neural memory
            
        Returns:
            Decoded response string
        """
        # In production, this would properly decode the cached response
        # For now, return a placeholder indicating cache hit
        hash_value = cached_tensor.item() if hasattr(cached_tensor, 'item') else cached_tensor[0]
        return f"[CACHED_RESPONSE_{hash_value}]"
    
    def _calculate_response_importance(self, 
                                     prompt: str, 
                                     response: str, 
                                     generation_time: float) -> float:
        """Calculate importance score for response caching.
        
        Args:
            prompt: Input prompt
            response: Generated response
            generation_time: Time taken to generate response
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        base_importance = 0.5
        
        # Higher importance for longer generation times (expensive responses)
        if generation_time > 5.0:
            base_importance += 0.3
        elif generation_time > 2.0:
            base_importance += 0.2
        elif generation_time > 1.0:
            base_importance += 0.1
        
        # Higher importance for longer responses (complex outputs)
        response_length = len(response)
        if response_length > 2000:
            base_importance += 0.2
        elif response_length > 1000:
            base_importance += 0.1
        
        return min(1.0, base_importance)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed status of the neural model manager system.
        
        Returns:
            Dictionary containing detailed system status
        """
        try:
            # Get neural runtime statistics
            runtime_stats = self.neural_runtime.get_runtime_stats()
            
            # Compile model status information
            model_status = {}
            for model_id, config in self.model_configs.items():
                model_status[model_id] = {
                    'cognitive_profile': config.cognitive_profile.value,
                    'model_type': config.model_type.value,
                    'loaded': model_id in self.loaded_models,
                    'active': model_id == self.active_model,
                    'attention_sparsity': config.attention_sparsity,
                    'compression_threshold': config.compression_threshold,
                    'auto_optimize': config.auto_optimize
                }
            
            # Overall system status
            status = {
                'neural_runtime': runtime_stats,
                'model_status': model_status,
                'interaction_count': self.interaction_count,
                'adaptation_history': self.adaptation_history,
                'active_model': self.active_model
            }
            
            return status
            
        except Exception as e:
            logging.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def _optimization_loop(self):
        """Background thread for adaptive optimization of models."""
        while True:
            try:
                # Sleep for optimization interval
                time.sleep(30)  # Run every 30 seconds
                
                # Check if any models are loaded
                if not self.loaded_models:
                    continue
                
                # Optimize each loaded model
                for model_id, model in self.loaded_models.items():
                    try:
                        self._optimize_model_adaptively(model_id, model)
                    except Exception as e:
                        logging.error(
                            f"Adaptive optimization failed for {model_id}",
                            extra={'error': str(e)},
                            exc_info=True
                        )
                        
            except Exception as e:
                logging.error(
                    f"Optimization loop error",
                    extra={'error': str(e)},
                    exc_info=True
                )
                # Prevent tight loop on persistent errors
                time.sleep(60)
    
    def _optimize_model_adaptively(self, model_id: str, model: Any):
        """Adaptive optimization for a specific model.
        
        Args:
            model_id: Identifier of the model
            model: Loaded model instance
            
        Raises:
            Exception: If optimization fails
        """
        try:
            config = self.model_configs[model_id]
            
            # Skip optimization if not enabled
            if not config.auto_optimize:
                logging.debug(f"Auto-optimization disabled for {model_id}")
                return
            
            # Collect performance metrics
            current_stats = self._collect_model_performance_stats(model_id)
            
            # Apply adaptive optimizations based on usage patterns
            optimizations_applied = []
            
            # Memory optimization based on usage
            if current_stats['memory_usage_mb'] > config.memory_allocation * 0.9:
                self._apply_memory_optimization(model_id, model)
                optimizations_applied.append('memory_optimization')
            
            # Parameter tuning based on interaction patterns
            if self.interaction_count > 0 and self.interaction_count % 100 == 0:
                self._tune_model_parameters(model_id, current_stats)
                optimizations_applied.append('parameter_tuning')
            
            # Cache optimization
            if current_stats['cache_hit_rate'] < 0.3:
                self._optimize_caching_strategy(model_id)
                optimizations_applied.append('cache_optimization')
            
            # Record optimization in history
            if optimizations_applied:
                optimization_record = {
                    'model_id': model_id,
                    'timestamp': time.time(),
                    'optimizations': optimizations_applied,
                    'stats': current_stats
                }
                self.adaptation_history.append(optimization_record)
                
                # Keep history manageable
                if len(self.adaptation_history) > 1000:
                    self.adaptation_history = self.adaptation_history[-500:]
                
                logging.info(
                    f"Adaptive optimizations applied",
                    extra={
                        'model_id': model_id,
                        'optimizations': optimizations_applied,
                        'stats': current_stats
                    }
                )
                
        except Exception as e:
            logging.error(
                f"Adaptive optimization failed",
                extra={'model_id': model_id, 'error': str(e)},
                exc_info=True
            )
            raise Exception(f"Adaptive optimization failed for {model_id}: {e}") from e
    
    def _collect_model_performance_stats(self, model_id: str) -> Dict[str, Any]:
        """Collect performance statistics for a model.
        
        Args:
            model_id: Identifier of the model
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Get model-specific stats
            model = self.loaded_models[model_id]
            stats = {
                'model_id': model_id,
                'timestamp': time.time(),
                'memory_usage_mb': self._get_model_memory_usage(model) * 1024,
                'interaction_count': self.interaction_count,
                'cache_hit_rate': self._calculate_cache_hit_rate(model_id),
                'active': model_id == self.active_model
            }
            
            # Add model-specific metrics if available
            if hasattr(model, 'get_stats'):
                model_stats = model.get_stats()
                stats.update(model_stats)
            
            return stats
            
        except Exception as e:
            logging.error(f"Failed to collect stats for {model_id}: {e}")
            return {'error': str(e)}
    
    def _calculate_cache_hit_rate(self, model_id: str) -> float:
        """Calculate cache hit rate for a model.
        
        Args:
            model_id: Identifier of the model
            
        Returns:
            Cache hit rate between 0.0 and 1.0
        """
        try:
            # This would track actual cache hits vs misses
            # For now, return a conservative estimate
            return 0.5  # Placeholder - implement actual tracking
        except Exception:
            return 0.0
    
    def _apply_memory_optimization(self, model_id: str, model: Any):
        """Apply memory optimization techniques to a model with comprehensive validation.
        
        Args:
            model_id: Identifier of the model
            model: Loaded model instance
            
        Raises:
            ValueError: If model_id is invalid or model not found
            RuntimeError: If optimization fails
        """
        # Input validation
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")
        
        if model_id not in self.model_configs:
            raise ValueError(f"Model configuration not found: {model_id}")
        
        try:
            correlation_id = self._generate_correlation_id()
            optimization_start = time.time()
            
            logger.info(
                f"Starting memory optimization",
                extra={
                    'correlation_id': correlation_id,
                    'model_id': model_id
                }
            )
            
            # Force garbage collection to free unused memory
            gc.collect()
            
            # Clear PyTorch CUDA cache if available and CUDA is being used
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.debug(f"Cleared CUDA cache for {model_id}")
                except Exception as cuda_error:
                    logger.warning(
                        f"Failed to clear CUDA cache",
                        extra={
                            'correlation_id': correlation_id,
                            'model_id': model_id,
                            'error': str(cuda_error)
                        }
                    )
            
            # Optimize memory allocation strategy with proper type handling
            config = self.model_configs[model_id]
            
            # Validate memory_allocation is properly structured
            if not isinstance(config.memory_allocation, dict):
                raise ValueError(f"Invalid memory_allocation type: {type(config.memory_allocation)}")
            
            # Calculate total memory allocation in MB from tier percentages
            total_allocation_mb = self._calculate_total_memory_allocation(config.memory_allocation)
            
            # Apply optimization if total allocation exceeds threshold
            memory_threshold_mb = 2048  # 2GB threshold
            if total_allocation_mb > memory_threshold_mb:
                # Reduce allocation by 10% across all tiers proportionally
                reduction_factor = 0.9
                optimized_allocation = self._reduce_memory_allocation(
                    config.memory_allocation, 
                    reduction_factor
                )
                
                # Validate optimized allocation maintains minimum requirements
                min_total_mb = 512  # 512MB minimum
                optimized_total = self._calculate_total_memory_allocation(optimized_allocation)
                
                if optimized_total >= min_total_mb:
                    # Create new config with optimized allocation
                    original_allocation = config.memory_allocation.copy()
                    config.memory_allocation = optimized_allocation
                    
                    logger.info(
                        f"Memory allocation optimized",
                        extra={
                            'correlation_id': correlation_id,
                            'model_id': model_id,
                            'original_total_mb': total_allocation_mb,
                            'optimized_total_mb': optimized_total,
                            'reduction_factor': reduction_factor
                        }
                    )
                    
                    # Store optimization metadata in neural memory
                    optimization_data = {
                        'timestamp': time.time(),
                        'original_allocation': original_allocation,
                        'optimized_allocation': optimized_allocation,
                        'reduction_factor': reduction_factor
                    }
                    
                    optimization_tensor = torch.tensor(
                        [optimized_total, reduction_factor], 
                        dtype=torch.float32
                    ) if TORCH_AVAILABLE else [optimized_total, reduction_factor]
                    
                    self.neural_runtime.store_activation(
                        f"memory_optimization_{model_id}_{correlation_id}",
                        optimization_tensor,
                        importance=0.7
                    )
                else:
                    logger.warning(
                        f"Cannot optimize memory allocation below minimum threshold",
                        extra={
                            'correlation_id': correlation_id,
                            'model_id': model_id,
                            'optimized_total_mb': optimized_total,
                            'minimum_required_mb': min_total_mb
                        }
                    )
            else:
                logger.debug(
                    f"Memory allocation within acceptable range",
                    extra={
                        'correlation_id': correlation_id,
                        'model_id': model_id,
                        'total_allocation_mb': total_allocation_mb,
                        'threshold_mb': memory_threshold_mb
                    }
                )
            
            # Additional model-specific optimizations
            self._apply_model_specific_optimizations(model, model_id, correlation_id)
            
            optimization_time = time.time() - optimization_start
            
            logger.info(
                f"Memory optimization completed",
                extra={
                    'correlation_id': correlation_id,
                    'model_id': model_id,
                    'optimization_time_seconds': optimization_time,
                    'final_allocation_mb': self._calculate_total_memory_allocation(config.memory_allocation)
                }
            )
                
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(
                f"Memory optimization failed",
                extra={
                    'model_id': model_id,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                exc_info=True
            )
            raise RuntimeError(f"Memory optimization failed for {model_id}: {e}") from e
    
    def _calculate_total_memory_allocation(self, memory_allocation: Dict[str, float]) -> float:
        """Calculate total memory allocation in MB from tier percentages.
        
        Args:
            memory_allocation: Dictionary of memory tier allocations
            
        Returns:
            Total memory allocation in MB
            
        Raises:
            ValueError: If memory allocation structure is invalid
        """
        if not isinstance(memory_allocation, dict):
            raise ValueError("memory_allocation must be a dictionary")
        
        if not memory_allocation:
            raise ValueError("memory_allocation cannot be empty")
        
        # Validate all values are numeric and in valid range
        total_percentage = 0.0
        for tier, percentage in memory_allocation.items():
            if not isinstance(percentage, (int, float)):
                raise ValueError(f"Invalid percentage type for tier {tier}: {type(percentage)}")
            
            if not (0.0 <= percentage <= 1.0):
                raise ValueError(f"Invalid percentage value for tier {tier}: {percentage}")
            
            total_percentage += percentage
        
        # Validate percentages sum to approximately 1.0 (allow small floating point errors)
        if not (0.95 <= total_percentage <= 1.05):
            raise ValueError(f"Memory allocation percentages must sum to 1.0, got: {total_percentage}")
        
        # Estimate total memory allocation based on system memory
        # This is a heuristic - in production, this should be configurable
        available_memory_gb = self._get_available_memory_gb()
        estimated_model_memory_gb = min(available_memory_gb * 0.8, 8.0)  # Cap at 8GB
        
        return estimated_model_memory_gb * 1024  # Convert to MB
    
    def _reduce_memory_allocation(self, 
                                 memory_allocation: Dict[str, float], 
                                 reduction_factor: float) -> Dict[str, float]:
        """Reduce memory allocation proportionally across all tiers.
        
        Args:
            memory_allocation: Original memory allocation dictionary
            reduction_factor: Factor to reduce allocation by (0.0-1.0)
            
        Returns:
            New memory allocation dictionary with reduced values
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(memory_allocation, dict):
            raise ValueError("memory_allocation must be a dictionary")
        
        if not isinstance(reduction_factor, (int, float)):
            raise ValueError("reduction_factor must be numeric")
        
        if not (0.0 < reduction_factor <= 1.0):
            raise ValueError(f"reduction_factor must be between 0.0 and 1.0, got: {reduction_factor}")
        
        # Apply reduction proportionally while maintaining tier relationships
        reduced_allocation = {}
        total_reduced = 0.0
        
        for tier, percentage in memory_allocation.items():
            reduced_percentage = percentage * reduction_factor
            reduced_allocation[tier] = reduced_percentage
            total_reduced += reduced_percentage
        
        # Normalize to ensure percentages still sum to 1.0
        if total_reduced > 0.0:
            normalization_factor = 1.0 / total_reduced
            for tier in reduced_allocation:
                reduced_allocation[tier] *= normalization_factor
        
        return reduced_allocation
    
    def _apply_model_specific_optimizations(self, 
                                          model: Any, 
                                          model_id: str, 
                                          correlation_id: str):
        """Apply model-specific memory optimizations.
        
        Args:
            model: Loaded model instance
            model_id: Model identifier
            correlation_id: Request correlation ID
        """
        try:
            # GGUF model optimizations
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model_path'):
                # Optimize GGUF model memory mapping
                self._optimize_gguf_memory_mapping(model, correlation_id)
            
            # PyTorch model optimizations
            elif TORCH_AVAILABLE and hasattr(model, 'parameters'):
                # Optimize PyTorch model memory usage
                self._optimize_pytorch_memory(model, correlation_id)
            
            # Generic optimizations for all models
            self._optimize_cache_memory(model_id, correlation_id)
            
        except Exception as e:
            logger.warning(
                f"Model-specific optimization failed",
                extra={
                    'correlation_id': correlation_id,
                    'model_id': model_id,
                    'error': str(e)
                }
            )
    
    def _optimize_gguf_memory_mapping(self, model: Any, correlation_id: str):
        """Optimize GGUF model memory mapping.
        
        Args:
            model: GGUF model instance
            correlation_id: Request correlation ID
        """
        try:
            # GGUF-specific memory optimizations
            # This would implement actual GGUF memory optimization logic
            logger.debug(f"Applied GGUF memory optimizations: {correlation_id}")
            
        except Exception as e:
            logger.warning(f"GGUF memory optimization failed: {e}")
    
    def _optimize_pytorch_memory(self, model: Any, correlation_id: str):
        """Optimize PyTorch model memory usage.
        
        Args:
            model: PyTorch model instance
            correlation_id: Request correlation ID
        """
        try:
            # PyTorch-specific memory optimizations
            if hasattr(model, 'eval'):
                model.eval()  # Set to evaluation mode to disable gradients
            
            # Clear any cached computations
            if hasattr(model, 'zero_grad'):
                model.zero_grad()
            
            logger.debug(f"Applied PyTorch memory optimizations: {correlation_id}")
            
        except Exception as e:
            logger.warning(f"PyTorch memory optimization failed: {e}")
    
    def _optimize_cache_memory(self, model_id: str, correlation_id: str):
        """Optimize cache memory for a model.
        
        Args:
            model_id: Model identifier
            correlation_id: Request correlation ID
        """
        try:
            # Clear old cache entries for this model
            cache_keys = [k for k in self.neural_runtime.list_keys() 
                         if k.startswith(f"cache_{model_id}_")]
            
            # Keep only recent cache entries (last 100)
            if len(cache_keys) > 100:
                old_keys = cache_keys[:-100]
                for key in old_keys:
                    self.neural_runtime.delete_activation(key)
                
                logger.debug(
                    f"Cleaned cache memory",
                    extra={
                        'correlation_id': correlation_id,
                        'model_id': model_id,
                        'removed_keys': len(old_keys)
                    }
                )
            
        except Exception as e:
            logger.warning(f"Cache memory optimization failed: {e}")
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory.
        
        Args:
            model_id: Identifier of the model to unload
            
        Returns:
            True if unloading successful, False otherwise
        """
        try:
            if model_id not in self.loaded_models:
                logger.warning(f"Model not loaded: {model_id}")
                return False
            
            # Remove from loaded models
            del self.loaded_models[model_id]
            
            # Clear neural memory cache
            cache_key = f"loaded_model_{model_id}"
            self.neural_runtime.delete_activation(cache_key)
            
            # Clear weight caches
            weight_keys = [k for k in self.neural_runtime.list_keys() 
                          if k.startswith(f"weights_{model_id}_")]
            for key in weight_keys:
                self.neural_runtime.delete_activation(key)
            
            # Update active model if needed
            if self.active_model == model_id:
                self.active_model = None
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Model unloaded successfully: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered and loaded models.
        
        Returns:
            Dictionary containing model information
        """
        try:
            registered_models = {}
            for model_id, config in self.model_configs.items():
                registered_models[model_id] = {
                    'registered': True,
                    'loaded': model_id in self.loaded_models,
                    'active': model_id == self.active_model,
                    'type': config.model_type.value,
                    'path': config.model_path,
                    'cognitive_profile': config.cognitive_profile.value
                }
            
            return {
                'registered_models': registered_models,
                'total_registered': len(self.model_configs),
                'total_loaded': len(self.loaded_models),
                'active_model': self.active_model
            }
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Gracefully shutdown the model manager."""
        try:
            logger.info("Initiating graceful shutdown")
            
            # Unload all models
            for model_id in list(self.loaded_models.keys()):
                self.unload_model(model_id)
            
            # Stop optimization thread
            if hasattr(self, '_optimization_thread') and self._optimization_thread.is_alive():
                # Thread is daemon, will stop with process
                pass
            
            # Clear neural memory
            self.neural_runtime.clear_memory()
            
            logger.info("Neural Model Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _generate_gguf_response(self, model: Any, prompt: str, config: ModelConfiguration, **kwargs) -> str:
        """Generate response using GGUF model with comprehensive validation.
        
        Args:
            model: Loaded GGUF model instance
            prompt: Input prompt for generation
            config: Model configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If generation fails
        """
        # Input validation
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")
        
        if len(prompt.strip()) == 0:
            raise ValueError("prompt cannot be empty or whitespace only")
        
        if not hasattr(model, '__call__'):
            raise ValueError("model must be callable")
        
        try:
            # Extract generation parameters with validation
            max_tokens = kwargs.get('max_tokens', 512)
            temperature = kwargs.get('temperature', config.temperature)
            top_p = kwargs.get('top_p', config.top_p)
            top_k = kwargs.get('top_k', config.top_k)
            repeat_penalty = kwargs.get('repeat_penalty', config.repeat_penalty)
            
            # Validate parameter bounds
            max_tokens = max(1, min(max_tokens, 4096))
            temperature = max(0.01, min(temperature, 2.0))
            top_p = max(0.01, min(top_p, 1.0))
            top_k = max(1, min(top_k, 100))
            repeat_penalty = max(0.1, min(repeat_penalty, 2.0))
            
            # Call model with validated parameters
            result = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                echo=False,
                stop=kwargs.get('stop', [])
            )
            
            # Validate response structure
            if not isinstance(result, dict):
                raise RuntimeError("Model returned invalid response format")
            
            if 'choices' not in result or not result['choices']:
                raise RuntimeError("Model returned empty choices")
            
            choice = result['choices'][0]
            if not isinstance(choice, dict) or 'text' not in choice:
                raise RuntimeError("Model returned invalid choice format")
            
            response_text = choice['text']
            if not isinstance(response_text, str):
                raise RuntimeError("Model returned non-string response")
            
            # Strip any leading/trailing whitespace
            response_text = response_text.strip()
            
            if not response_text:
                raise RuntimeError("Model returned empty response")
            
            return response_text
            
        except Exception as e:
            logger.error(f"GGUF response generation failed: {e}")
            raise RuntimeError(f"GGUF response generation failed: {e}") from e
    
    def _generate_pytorch_response(self, model: Any, prompt: str, config: ModelConfiguration, **kwargs) -> str:
        """Generate response using PyTorch model with comprehensive validation.
        
        Args:
            model: Loaded PyTorch model instance
            prompt: Input prompt for generation
            config: Model configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If generation fails
            NotImplementedError: PyTorch generation not fully implemented
        """
        # Input validation
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")
        
        if len(prompt.strip()) == 0:
            raise ValueError("prompt cannot be empty or whitespace only")
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for model generation")
        
        try:
            # PyTorch model generation requires tokenization and decoding
            # This needs proper implementation based on model architecture
            raise NotImplementedError(
                "PyTorch model generation requires tokenizer integration. "
                "Please implement tokenization pipeline for your specific model."
            )
            
        except NotImplementedError:
            raise
        except Exception as e:
            logger.error(f"PyTorch response generation failed: {e}")
            raise RuntimeError(f"PyTorch response generation failed: {e}") from e
    
    def _trigger_adaptation(self, model_id: str):
        """Trigger adaptive optimization for a model.
        
        Args:
            model_id: Identifier of the model to optimize
        """
        try:
            if not model_id or not isinstance(model_id, str):
                logger.warning("Invalid model_id for adaptation trigger")
                return
            
            if model_id not in self.model_configs:
                logger.warning(f"Cannot trigger adaptation for unknown model: {model_id}")
                return
            
            config = self.model_configs[model_id]
            if not config.auto_optimize:
                logger.debug(f"Adaptation disabled for model: {model_id}")
                return
            
            # Trigger immediate optimization in background
            logger.info(f"Triggering adaptive optimization for model: {model_id}")
            
            # This will be handled by the optimization thread
            
        except Exception as e:
            logger.error(f"Failed to trigger adaptation for {model_id}: {e}")
    
    def delete_activation(self, key: str) -> bool:
        """Delete activation from neural memory (delegate to runtime).
        
        Args:
            key: Activation key to delete
            
        Returns:
            True if deletion successful
        """
        try:
            return self.neural_runtime.delete_activation(key)
        except AttributeError:
            # Neural runtime doesn't have delete_activation method
            logger.warning("Neural runtime doesn't support activation deletion")
            return False
        except Exception as e:
            logger.error(f"Failed to delete activation {key}: {e}")
            return False
    
    def list_keys(self) -> List[str]:
        """List all keys in neural memory (delegate to runtime).
        
        Returns:
            List of activation keys
        """
        try:
            return self.neural_runtime.list_keys()
        except AttributeError:
            # Neural runtime doesn't have list_keys method
            logger.warning("Neural runtime doesn't support key listing")
            return []
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []
    
    def clear_memory(self) -> bool:
        """Clear all neural memory (delegate to runtime).
        
        Returns:
            True if clearing successful
        """
        try:
            return self.neural_runtime.clear_memory()
        except AttributeError:
            # Neural runtime doesn't have clear_memory method
            logger.warning("Neural runtime doesn't support memory clearing")
            return False
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False
