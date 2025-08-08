"""
GPT-Ø System Launcher and Terminal User Interface
=================================================

Production-grade launcher for the GPT-Ø self-modifying multimodal AI system.
Provides comprehensive logging, health monitoring, colored ASCII TUI, and
extensive debugging capabilities for system development and operation.

Features:
- Colored ASCII terminal interface with real-time status
- Comprehensive error handling and logging with correlation IDs
- Health monitoring and performance metrics
- Interactive debugging and introspection
- Graceful shutdown and resource cleanup
- Configuration validation and auto-recovery
- Memory usage monitoring and optimization

Author: Morpheus
Date: August 3, 2025
Version: 2.0.0 - Production Release
"""

import sys
import os
import json
import time
import signal
import logging
import threading
import traceback
import argparse
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import uuid
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

# Rich imports for colored terminal interface
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.tree import Tree
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich.status import Status
from rich.logging import RichHandler

# System imports
import asyncio
import argparse
import traceback
import threading
import logging
import signal
import time
import json
import os
import sys
from gpt_model import GPT_Ø, ModalityType, ChainStability
from tokenizer_adapter import TokenizerAdapter
from bayesian_config_orchestrator import BayesianConfigurationOrchestrator
from cas.neural_memory_runtime import NeuralMemoryRuntime, integrate_neural_memory_runtime
from recursive_weights_core import RecursiveWeightRegistry, RecursiveWeightConfig
from tool_output_head import UniversalToolControlOutputHead, UniversalControlConfig, EclogueConfig

# Exception handling
from exceptions import (
    GPTModelError, ConfigurationError, InitializationError,
    GenerationError, ModalityError, ReasoningError, MemoryError, SafetyError
)

# Configure structured logging with correlation IDs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s',
    handlers=[
        RichHandler(console=Console(stderr=True), rich_tracebacks=True),
        logging.FileHandler('gpt_zero_system.log', encoding='utf-8')
    ]
)

class CorrelationIdFilter(logging.Filter):
    """Adds correlation IDs to all log records for traceability."""
    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = getattr(threading.current_thread(), 'correlation_id', 'MAIN')
        return True

# Apply filter to all handlers
for handler in logging.root.handlers:
    handler.addFilter(CorrelationIdFilter())

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"

class InteractionMode(Enum):
    """User interaction modes."""
    CHAT = "chat"
    DEBUG = "debug"
    MONITOR = "monitor"
    TOOLS = "tools"
    CONFIG = "config"

@dataclass
class SystemMetrics:
    """Real-time system performance metrics."""
    cpu_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    torch_memory_allocated_mb: float = 0.0
    torch_memory_cached_mb: float = 0.0
    active_threads: int = 0
    reasoning_chains_active: int = 0
    reasoning_chains_completed: int = 0
    total_tokens_processed: int = 0
    errors_last_hour: int = 0
    uptime_seconds: float = 0.0
    last_updated: float = field(default_factory=time.time)

@dataclass
class ConfigurationState:
    """System configuration state and validation."""
    config_path: Path
    config_valid: bool = False
    config_data: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = field(default_factory=list)
    last_modified: float = 0.0
    auto_reload: bool = True

class GPTZeroLauncher:
    """
    Production-grade launcher and TUI for the GPT-Ø system.
    
    Provides comprehensive system management, monitoring, and interaction
    capabilities with extensive error handling and debugging support.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the GPT-Ø launcher with comprehensive setup.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            InitializationError: If critical initialization fails
        """
        # Set correlation ID for main thread
        threading.current_thread().correlation_id = str(uuid.uuid4())[:8]
        
        self.console = Console()
        self.system_state = SystemState.INITIALIZING
        self.interaction_mode = InteractionMode.CHAT
        self.start_time = time.time()
        self.shutdown_requested = False
        
        # Initialize paths and configuration
        self.project_root = Path(__file__).parent
        self.config_path = Path(config_path) if config_path else self.project_root / "config" / "agent_config.json"
        self.log_path = self.project_root / "logs"
        self.log_path.mkdir(exist_ok=True)
        
        # System components (initialized later)
        self.gpt_model: Optional[GPT_Ø] = None
        self.tokenizer: Optional[TokenizerAdapter] = None
        self.config_orchestrator: Optional[BayesianConfigurationOrchestrator] = None
        self.neural_memory_runtime: Optional[NeuralMemoryRuntime] = None
        self.recursive_weights_registry: Optional[RecursiveWeightRegistry] = None
        self.tool_output_head: Optional[UniversalToolControlOutputHead] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Monitoring and metrics
        self.metrics = SystemMetrics()
        self.config_state = ConfigurationState(config_path=self.config_path)
        self.error_history: List[Dict[str, Any]] = []
        self.performance_history: List[SystemMetrics] = []
        
        # Threading and shutdown handling
        self.monitor_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("GPT-Ø Launcher initialized", extra={'correlation_id': 'LAUNCHER_INIT'})

    def _initialize_system_components(self) -> bool:
        """
        Initialize all system components with comprehensive error handling.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            with self.console.status("[bold green]Initializing system components...") as status:
                
                # Initialize device
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Using device: {self.device}")
                status.update(f"[bold green]Device selected: {self.device}")
                time.sleep(0.5)
                
                # Step 1: Initialize configuration orchestrator
                status.update("[bold green]Initializing Bayesian configuration orchestrator...")
                try:
                    self.config_orchestrator = BayesianConfigurationOrchestrator(
                        config_path=str(self.config_path)
                    )
                    logger.info("BayesianConfigurationOrchestrator initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize configuration orchestrator: {e}", exc_info=True)
                    raise InitializationError("Configuration orchestrator failed to load") from e

                
                # Step 2: Initialize recursive weights registry  
                status.update("[bold green]Initializing recursive weights system...")
                try:
                    recursive_config = RecursiveWeightConfig(
                        max_recursion_depth=self.config_orchestrator.get_parameter_value('recursive_weights.max_recursion_depth', 5),
                        convergence_threshold=self.config_orchestrator.get_parameter_value('recursive_weights.convergence_threshold', 1e-6),
                        cache_size=1000
                    )
                    self.recursive_weights_registry = RecursiveWeightRegistry(config=recursive_config)
                    logger.info("RecursiveWeightRegistry initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize recursive weights registry: {e}", exc_info=True)
                    raise InitializationError("Recursive weights registry failed to load") from e

                
                # Step 3: Initialize neural memory runtime
                status.update("[bold green]Initializing neural memory runtime...")
                try:
                    memory_config = {
                        'max_memory_gb': self.config_orchestrator.get_parameter_value('neural_memory_runtime.max_memory_gb', 6.0),
                        'd_model': self.config_orchestrator.get_parameter_value('model_params.d_model'),
                        'n_heads': self.config_orchestrator.get_parameter_value('model_params.n_heads'),
                        'sparsity_ratio': self.config_orchestrator.get_parameter_value('neural_memory_runtime.sparsity_ratio', 0.1),
                        'summary_ratio': self.config_orchestrator.get_parameter_value('neural_memory_runtime.summary_ratio', 0.1),
                        'enable_neural_memory': self.config_orchestrator.get_parameter_value('neural_memory_runtime.enable_neural_memory', True)
                    }
                    self.neural_memory_runtime = NeuralMemoryRuntime(config=memory_config)
                    logger.info("NeuralMemoryRuntime initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize neural memory runtime: {e}", exc_info=True)
                    raise InitializationError("Neural memory runtime failed to load") from e

                
                # Step 4: Initialize tokenizer adapter
                status.update("[bold green]Initializing tokenizer adapter...")
                try:
                    self.tokenizer = TokenizerAdapter(config_path=self.config_path)
                    logger.info("TokenizerAdapter initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize tokenizer: {e}", exc_info=True)
                    raise InitializationError("Tokenizer failed to load") from e

                
                # Step 5: Initialize GPT-Ø model
                status.update("[bold green]Initializing GPT-Ø model...")
                try:
                    self.gpt_model = GPT_Ø(
                        config_path=str(self.config_path)
                    ).to(self.device)
                    
                    # Integrate neural memory runtime with the model
                    if self.neural_memory_runtime:
                        integrate_neural_memory_runtime(self.gpt_model, memory_config)
                        logger.info("Neural memory runtime integrated with GPT-Ø model")
                    
                    # Verify model initialization
                    if not hasattr(self.gpt_model, 'token_embeddings'):
                        raise InitializationError("Model missing token embeddings")
                    
                    logger.info("GPT-Ø model initialized successfully")
                    logger.info(f"Model parameters: vocab_size={self.gpt_model.vocab_size}, "
                              f"d_model={self.gpt_model.d_model}, n_layers={self.gpt_model.n_layers}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize GPT-Ø model: {e}", exc_info=True)
                    raise InitializationError("GPT-Ø model failed to load") from e

                
                # Step 6: Initialize tool output head
                status.update("[bold green]Initializing universal tool control head...")
                try:
                    # Create EclogueConfig for tool head
                    tool_config = EclogueConfig(
                        hidden_size=self.config_orchestrator.get_parameter_value('model_params.d_model'),
                        num_attention_heads=self.config_orchestrator.get_parameter_value('model_params.n_heads'),
                        num_layers=4,  # Tool head specific layers
                        vocab_size=self.config_orchestrator.get_parameter_value('model_params.vocab_size'),
                        max_position_embeddings=self.config_orchestrator.get_parameter_value('model_params.max_seq_len')
                    )
                    
                    self.tool_output_head = UniversalToolControlOutputHead(
                        config=tool_config
                    ).to(self.device)
                    
                    logger.info("UniversalToolControlOutputHead initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize tool output head: {e}", exc_info=True)
                    raise InitializationError("Tool output head failed to load") from e

                
                # Step 7: Initialize monitoring
                status.update("[bold green]Starting system monitoring...")
                self._start_monitoring()
                
                # Update system state
                self.system_state = SystemState.READY
                logger.info("All system components initialized successfully")
                
                # Display successful initialization summary
                self.console.print("\n[bold green]=> System Initialization Complete[/bold green]")
                init_table = Table(title="Initialized Components", show_header=True, header_style="bold blue")
                init_table.add_column("Component", style="cyan")
                init_table.add_column("Status", style="green")
                init_table.add_column("Details", style="yellow")
                
                init_table.add_row("BayesianConfigurationOrchestrator", "-> Ready", "Parameter optimization enabled")
                init_table.add_row("RecursiveWeightRegistry", "-> Ready", f"Max depth: {recursive_config.max_recursion_depth}")
                init_table.add_row("NeuralMemoryRuntime", "-> Ready", f"Memory limit: {memory_config['max_memory_gb']} GB")
                init_table.add_row("TokenizerAdapter", "-> Ready", f"Vocab size: {self.tokenizer.vocab_size}")
                init_table.add_row("GPT-Ø Model", "-> Ready", f"{self.gpt_model.d_model}d, {self.gpt_model.n_layers}L, {self.device}")
                init_table.add_row("UniversalToolControlOutputHead", "-> Ready", "Tool synthesis enabled")
                
                self.console.print(init_table)
                
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}", exc_info=True)
            self.system_state = SystemState.ERROR
            self.console.print(f"[bold red]=> Initialization failed: {e}[/bold red]")
            return False

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown_requested = True
        self.shutdown_event.set()

    def _validate_environment(self) -> List[str]:
        """
        Validate system environment and dependencies.
        
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                errors.append(f"Python 3.8+ required, found {sys.version}")
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                errors.append("CUDA not available - GPU acceleration disabled")
            else:
                try:
                    device_count = torch.cuda.device_count()
                    logger.info(f"CUDA available with {device_count} device(s)")
                    for i in range(device_count):
                        device_name = torch.cuda.get_device_name(i)
                        memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
                        logger.info(f"GPU {i}: {device_name} ({memory_gb:.1f} GB)")
                except Exception as e:
                    errors.append(f"CUDA detection error: {e}")
            
            # Check memory requirements
            available_memory_gb = psutil.virtual_memory().available / 1e9
            if available_memory_gb < 8:
                errors.append(f"Insufficient RAM: {available_memory_gb:.1f} GB available, 8+ GB recommended")
            
            # Check disk space
            disk_free_gb = psutil.disk_usage(str(self.project_root)).free / 1e9
            if disk_free_gb < 5:
                errors.append(f"Low disk space: {disk_free_gb:.1f} GB free, 5+ GB recommended")
            
            # Validate configuration file exists
            if not self.config_path.exists():
                errors.append(f"Configuration file not found: {self.config_path}")
            
            # Check required directories
            required_dirs = ['config', 'logs', 'models']
            for dir_name in required_dirs:
                dir_path = self.project_root / dir_name
                if not dir_path.exists():
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created directory: {dir_path}")
                    except Exception as e:
                        errors.append(f"Cannot create directory {dir_path}: {e}")
            
        except Exception as e:
            errors.append(f"Environment validation failed: {e}")
            logger.error(f"Environment validation error: {e}", exc_info=True)
        
        return errors

    def _load_and_validate_config(self) -> bool:
        """
        Load and validate system configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                self.config_state.validation_errors = [f"File not found: {self.config_path}"]
                return False
            
            # Load configuration
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_state.config_data = json.load(f)
            
            self.config_state.last_modified = self.config_path.stat().st_mtime
            
            # Validate required sections
            required_sections = [
                'model_params', 'generation_params', 'reasoning_params',
                'neural_memory_runtime', 'recursive_weights'
            ]
            
            validation_errors = []
            for section in required_sections:
                if section not in self.config_state.config_data:
                    validation_errors.append(f"Missing required section: {section}")
            
            # Validate model parameters
            model_params = self.config_state.config_data.get('model_params', {})
            required_model_params = [
                'vocab_size', 'd_model', 'n_layers', 'n_heads',
                'max_seq_len', 'memory_capacity', 'dropout'
            ]
            
            for param in required_model_params:
                if param not in model_params:
                    validation_errors.append(f"Missing model parameter: {param}")
                elif not isinstance(model_params[param], (int, float)):
                    validation_errors.append(f"Invalid model parameter type: {param}")
            
            # Validate ranges
            if model_params.get('d_model', 0) % model_params.get('n_heads', 1) != 0:
                validation_errors.append("d_model must be divisible by n_heads")
            
            if model_params.get('vocab_size', 0) <= 0:
                validation_errors.append("vocab_size must be positive")
            
            if not (0.0 <= model_params.get('dropout', -1) <= 1.0):
                validation_errors.append("dropout must be between 0.0 and 1.0")
            
            self.config_state.validation_errors = validation_errors
            self.config_state.config_valid = len(validation_errors) == 0
            
            if self.config_state.config_valid:
                logger.info("Configuration validation successful")
            else:
                logger.error(f"Configuration validation failed: {validation_errors}")
            
            return self.config_state.config_valid
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration file: {e}"
            logger.error(error_msg)
            self.config_state.validation_errors = [error_msg]
            return False
        except Exception as e:
            error_msg = f"Configuration loading failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.config_state.validation_errors = [error_msg]
            return False

    def _initialize_system_components(self) -> bool:
        """
        Initialize all system components with comprehensive error handling.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            with self.console.status("[bold green]Initializing system components...") as status:
                
                # Initialize device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Using device: {device}")
                status.update(f"[bold green]Device selected: {device}")
                time.sleep(0.5)
                
                # Step 1: Initialize configuration orchestrator
                status.update("[bold green]Initializing Bayesian configuration orchestrator...")
                try:
                    self.config_orchestrator = BayesianConfigurationOrchestrator(
                        config_path=str(self.config_path)
                    )
                    logger.info("BayesianConfigurationOrchestrator initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize configuration orchestrator: {e}", exc_info=True)
                    return False
                
                # Step 2: Initialize recursive weights registry  
                status.update("[bold green]Initializing recursive weights system...")
                try:
                    recursive_config = RecursiveWeightConfig(
                        max_recursion_depth=self.config_state.config_data.get('recursive_weights', {}).get('max_recursion_depth', 5),
                        convergence_threshold=self.config_state.config_data.get('recursive_weights', {}).get('convergence_threshold', 1e-6),
                        cache_size=1000
                    )
                    self.recursive_weights_registry = RecursiveWeightRegistry(config=recursive_config)
                    logger.info("RecursiveWeightRegistry initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize recursive weights registry: {e}", exc_info=True)
                    return False
                
                # Step 3: Initialize neural memory runtime
                status.update("[bold green]Initializing neural memory runtime...")
                try:
                    memory_config = {
                        'max_memory_gb': 6.0,
                        'd_model': self.config_state.config_data['model_params']['d_model'],
                        'n_heads': self.config_state.config_data['model_params']['n_heads'],
                        'sparsity_ratio': 0.1,
                        'summary_ratio': 0.1,
                        'enable_neural_memory': self.config_state.config_data.get('neural_memory_runtime', {}).get('enable_neural_memory', True)
                    }
                    self.neural_memory_runtime = NeuralMemoryRuntime(config=memory_config)
                    logger.info("NeuralMemoryRuntime initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize neural memory runtime: {e}", exc_info=True)
                    return False
                
                # Step 4: Initialize tokenizer adapter
                status.update("[bold green]Initializing tokenizer adapter...")
                try:
                    self.tokenizer = TokenizerAdapter(config_path=self.config_path)
                    logger.info("TokenizerAdapter initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize tokenizer: {e}", exc_info=True)
                    return False
                
                # Step 5: Initialize GPT-Ø model
                status.update("[bold green]Initializing GPT-Ø model...")
                try:
                    model_params = self.config_state.config_data['model_params']
                    self.gpt_model = GPT_Ø(
                        config_file=str(self.config_path),
                        device=device
                    ).to(device)
                    
                    # Integrate neural memory runtime with the model
                    if self.neural_memory_runtime:
                        integrate_neural_memory_runtime(self.gpt_model, memory_config)
                        logger.info("Neural memory runtime integrated with GPT-Ø model")
                    
                    # Verify model initialization
                    if not hasattr(self.gpt_model, 'token_embeddings'):
                        raise InitializationError("Model missing token embeddings")
                    
                    logger.info("GPT-Ø model initialized successfully")
                    logger.info(f"Model parameters: vocab_size={self.gpt_model.vocab_size}, "
                              f"d_model={self.gpt_model.d_model}, n_layers={self.gpt_model.n_layers}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize GPT-Ø model: {e}", exc_info=True)
                    return False
                
                # Step 6: Initialize tool output head
                status.update("[bold green]Initializing universal tool control head...")
                try:
                    # Create EclogueConfig for tool head
                    tool_config = EclogueConfig(
                        d_model=self.config_state.config_data['model_params']['d_model'],
                        n_heads=self.config_state.config_data['model_params']['n_heads'],
                        n_layers=4,  # Tool head specific layers
                        dropout=0.1,
                        max_sequence_length=self.config_state.config_data['model_params']['max_seq_len']
                    )
                    
                    # Create UniversalControlConfig
                    control_config = UniversalControlConfig(
                        enable_tool_synthesis=True,
                        enable_external_control=True,
                        max_risk_level="medium",
                        safety_validation=True,
                        autonomous_mode=False
                    )
                    
                    self.tool_output_head = UniversalToolControlOutputHead(
                        config=tool_config,
                        control_config=control_config
                    ).to(device)
                    
                    logger.info("UniversalToolControlOutputHead initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize tool output head: {e}", exc_info=True)
                    return False
                
                # Step 7: Initialize monitoring
                status.update("[bold green]Starting system monitoring...")
                self._start_monitoring()
                
                # Update system state
                self.system_state = SystemState.READY
                logger.info("All system components initialized successfully")
                
                # Display successful initialization summary
                self.console.print("
[bold green]=> System Initialization Complete[/bold green]")
                init_table = Table(title="Initialized Components", show_header=True, header_style="bold blue")
                init_table.add_column("Component", style="cyan")
                init_table.add_column("Status", style="green")
                init_table.add_column("Details", style="yellow")
                
                init_table.add_row("BayesianConfigurationOrchestrator", "✓ Ready", "Parameter optimization enabled")
                init_table.add_row("RecursiveWeightRegistry", "✓ Ready", f"Max depth: {recursive_config.max_recursion_depth}")
                init_table.add_row("NeuralMemoryRuntime", "✓ Ready", f"Memory limit: {memory_config['max_memory_gb']} GB")
                init_table.add_row("TokenizerAdapter", "✓ Ready", f"Vocab size: {self.tokenizer.vocab_size}")
                init_table.add_row("GPT-Ø Model", "✓ Ready", f"{self.gpt_model.d_model}d, {self.gpt_model.n_layers}L, {device}")
                init_table.add_row("UniversalToolControlOutputHead", "✓ Ready", "Tool synthesis enabled")
                
                self.console.print(init_table)
                
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}", exc_info=True)
            self.system_state = SystemState.ERROR
            self.console.print(f"[bold red]❌ Initialization failed: {e}[/bold red]")
            return False

    def _start_monitoring(self) -> None:
        """Start background monitoring thread."""
        def monitor_loop():
            threading.current_thread().correlation_id = 'MONITOR'
            logger.info("System monitoring started")
            
            while not self.shutdown_event.is_set():
                try:
                    self._update_metrics()
                    time.sleep(1.0)  # Update every second
                except Exception as e:
                    logger.error(f"Monitoring error: {e}", exc_info=True)
                    time.sleep(5.0)  # Longer delay on error
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _update_metrics(self) -> None:
        """Update system performance metrics."""
        try:
            # System metrics
            self.metrics.cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.metrics.memory_usage_mb = memory.used / 1024 / 1024
            self.metrics.memory_percent = memory.percent
            self.metrics.active_threads = threading.active_count()
            self.metrics.uptime_seconds = time.time() - self.start_time
            
            # GPU metrics
            if torch.cuda.is_available():
                try:
                    self.metrics.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    self.metrics.torch_memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    self.metrics.torch_memory_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
                except Exception:
                    pass  # GPU metrics optional
            
            # Model-specific metrics
            if self.gpt_model:
                try:
                    cot_metrics = self.gpt_model.get_cot_performance_metrics()
                    self.metrics.reasoning_chains_active = len(self.gpt_model.get_active_reasoning_chains())
                    self.metrics.reasoning_chains_completed = cot_metrics.get('total_chains_processed', 0)
                    self.metrics.total_tokens_processed = cot_metrics.get('total_tokens_processed', 0)
                except Exception:
                    pass  # Model metrics optional
            
            # Error tracking
            self.metrics.errors_last_hour = len([
                error for error in self.error_history
                if time.time() - error['timestamp'] < 3600
            ])
            
            self.metrics.last_updated = time.time()
            
            # Store metrics history (keep last 1000 entries)
            self.performance_history.append(self.metrics)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            logger.error(f"Metrics update failed: {e}", exc_info=True)

    def _create_status_display(self) -> Layout:
        """Create rich layout for system status display."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Header
        header_text = Text("GPT-Ø System Control Interface", style="bold magenta")
        header_panel = Panel(
            Align.center(header_text),
            style="bright_blue"
        )
        layout["header"].update(header_panel)
        
        # System status
        status_table = Table(title="System Status", show_header=True, header_style="bold blue")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="yellow")
        
        # Add system status rows
        status_table.add_row(
            "System State",
            self.system_state.value.title(),
            f"Uptime: {self.metrics.uptime_seconds:.1f}s"
        )
        
        if self.gpt_model:
            status_table.add_row(
                "GPT-Ø Model",
                "✓ Ready" if self.system_state == SystemState.READY else "⚠ Loading",
                f"Params: {self.gpt_model.d_model}d, {self.gpt_model.n_layers}L"
            )
        
        status_table.add_row(
            "Memory",
            f"{self.metrics.memory_percent:.1f}%",
            f"{self.metrics.memory_usage_mb:.0f} MB"
        )
        
        if torch.cuda.is_available():
            status_table.add_row(
                "GPU",
                "✓ Available",
                f"{self.metrics.gpu_memory_mb:.0f} MB allocated"
            )
        
        layout["left"].update(Panel(status_table, title="Status", border_style="green"))
        
        # Performance metrics
        perf_table = Table(title="Performance", show_header=True, header_style="bold blue")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf_table.add_row("CPU Usage", f"{self.metrics.cpu_percent:.1f}%")
        perf_table.add_row("Active Threads", str(self.metrics.active_threads))
        perf_table.add_row("Reasoning Chains", f"{self.metrics.reasoning_chains_active} active")
        perf_table.add_row("Tokens Processed", str(self.metrics.total_tokens_processed))
        perf_table.add_row("Errors (1h)", str(self.metrics.errors_last_hour))
        
        layout["right"].update(Panel(perf_table, title="Metrics", border_style="blue"))
        
        # Footer with interaction help
        footer_text = Text()
        footer_text.append("Commands: ", style="bold")
        footer_text.append("chat", style="green")
        footer_text.append(" | ", style="white")
        footer_text.append("debug", style="yellow")
        footer_text.append(" | ", style="white")
        footer_text.append("monitor", style="blue")
        footer_text.append(" | ", style="white")
        footer_text.append("tools", style="magenta")
        footer_text.append(" | ", style="white")
        footer_text.append("quit", style="red")
        
        layout["footer"].update(Panel(
            Align.center(footer_text),
            style="bright_black"
        ))
        
        return layout

    async def _handle_chat_mode(self) -> None:
        """Handle interactive chat with the GPT-Ø model."""
        self.console.print("
[bold green]=> GPT-Ø Chat Mode[/bold green]")
        self.console.print("Multimodal AI with neural memory, recursive weights, and tool synthesis")
        self.console.print("Type 'back' to return to main menu, 'quit' to exit\n")
        
        # Initialize chat context
        conversation_history = []
        
        while not self.shutdown_requested:
            try:
                prompt = await self.console.input("You: ")
                
                if prompt.lower() in ['back', 'exit']:
                    break
                elif prompt.lower() == 'quit':
                    self.shutdown_requested = True
                    break
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": prompt})
                
                # Generate response with comprehensive error handling
                with self.console.status("[bold green]🧠 GPT-Ø is thinking...") as status:
                    try:
                        # Encode input using tokenizer adapter
                        input_ids = self.tokenizer.encode(prompt)
                        if not isinstance(input_ids, torch.Tensor):
                            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                        
                        # Prepare input for multimodal processing
                        input_data = {
                            'text': prompt,
                            'tokens': input_ids,
                            'conversation_history': conversation_history[-10:]  # Keep last 10 exchanges
                        }
                        
                        # Generate response using GPT-Ø
                        generation_params = self.config_orchestrator.get_parameter_value('generation_params', {})
                        
                        # Use the model's generate method with full context
                        result = self.gpt_model.generate(
                            input_data=input_data,
                            modality=ModalityType.TEXT,
                            max_length=generation_params.get('max_length', 256),
                            temperature=generation_params.get('temperature', 0.8),
                            top_k=generation_params.get('top_k', 40),
                            top_p=generation_params.get('top_p', 0.9)
                        )
                        # Process and display response
                        if result and 'generated_tokens' in result:
                            # Decode response using tokenizer
                            response_tokens = result['generated_tokens']
                            if isinstance(response_tokens, torch.Tensor):
                                response_text = self.tokenizer.decode(response_tokens.cpu().tolist()[0])
                            else:
                                response_text = self.tokenizer.decode(response_tokens)
                            
                            # Clean up response text
                            response_text = response_text.strip()
                            
                            # Display response with rich formatting
                            self.console.print(f"\n[bold blue]GPT-Ø:[/bold blue] {response_text}")
                            
                            # Add to conversation history
                            conversation_history.append({"role": "assistant", "content": response_text})
                            
                        else:
                            self.console.print("[yellow]⚠ No response generated.[/yellow]")

                    except GenerationError as e:
                        logger.error(f"Generation error: {e}", exc_info=True)
                        self.console.print(f"[red]❌ Generation error: {e}[/red]")
                    except Exception as e:
                        logger.error(f"An unexpected error occurred during generation: {e}", exc_info=True)
                        self.console.print(f"[red]❌ An unexpected error occurred: {e}[/red]")

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                logger.error(f"Chat error: {e}", exc_info=True)
                self.console.print(f"[red]❌ Chat error: {e}[/red]")
                self._record_error("chat", str(e))

    def _handle_debug_mode(self) -> None:
        """Handle debug and introspection mode."""
        self.console.print("\n[bold yellow]Debug Mode[/bold yellow]")
        
        debug_commands = {
            'introspect': self._debug_introspect,
            'memory': self._debug_memory,
            'reasoning': self._debug_reasoning,
            'config': self._debug_config,
            'logs': self._debug_logs,
            'torch': self._debug_torch,
            'back': lambda: None
        }
        
        while not self.shutdown_requested:
            self.console.print("\nDebug Commands:")
            for cmd in debug_commands.keys():
                self.console.print(f"  [cyan]{cmd}[/cyan]")
            
            command = Prompt.ask("Debug command", choices=list(debug_commands.keys()), default="back")
            
            if command == 'back':
                break
            
            try:
                debug_commands[command]()
            except Exception as e:
                logger.error(f"Debug command error: {e}", exc_info=True)
                self.console.print(f"[red]❌ Debug error: {e}[/red]")

    def _debug_introspect(self) -> None:
        """Debug introspection of model state."""
        if not self.gpt_model:
            self.console.print("[red]❌ Model not initialized[/red]")
            return
        
        try:
            with self.console.status("[bold blue]Gathering introspection data..."):
                # Get model introspection data
                introspection = {}
                
                # Basic model info
                introspection['model_info'] = {
                    'vocab_size': getattr(self.gpt_model, 'vocab_size', 'unknown'),
                    'd_model': getattr(self.gpt_model, 'd_model', 'unknown'),
                    'n_layers': getattr(self.gpt_model, 'n_layers', 'unknown'),
                    'n_heads': getattr(self.gpt_model, 'n_heads', 'unknown'),
                    'device': str(next(self.gpt_model.parameters()).device) if hasattr(self.gpt_model, 'parameters') else 'unknown'
                }
                
                # Memory system info
                if self.neural_memory_runtime:
                    memory_stats = self.neural_memory_runtime.get_runtime_stats()
                    introspection['memory_info'] = {
                        'tensor_memory_usage': memory_stats.get('memory_usage_mb', 0),
                        'tensor_memory_capacity': memory_stats.get('max_memory_mb', 0),
                        'memory_utilization_percent': memory_stats.get('utilization_percent', 0),
                        'average_memory_importance': memory_stats.get('avg_importance', 0),
                        'active_chunks': memory_stats.get('active_chunks', 0)
                    }
                else:
                    introspection['memory_info'] = {'status': 'not_initialized'}
                
                # Reasoning system info
                if hasattr(self.gpt_model, 'get_cot_performance_metrics'):
                    cot_metrics = self.gpt_model.get_cot_performance_metrics()
                    active_chains = getattr(self.gpt_model, 'get_active_reasoning_chains', lambda: [])()
                    introspection['reasoning_info'] = {
                        'enabled': True,
                        'active_chains_count': len(active_chains) if active_chains else 0,
                        'completed_chains_count': cot_metrics.get('total_chains_processed', 0),
                        'total_reasoning_steps': cot_metrics.get('total_reasoning_steps', 0),
                        'stability_violations': cot_metrics.get('stability_violations', 0)
                    }
                else:
                    introspection['reasoning_info'] = {'enabled': False, 'status': 'not_available'}
                
                # Architecture info
                if hasattr(self.gpt_model, 'parameters'):
                    total_params = sum(p.numel() for p in self.gpt_model.parameters())
                    introspection['architecture_info'] = {
                        'total_parameters': total_params,
                        'n_layers': getattr(self.gpt_model, 'n_layers', 'unknown'),
                        'n_heads': getattr(self.gpt_model, 'n_heads', 'unknown'),
                        'd_model': getattr(self.gpt_model, 'd_model', 'unknown')
                    }
                else:
                    introspection['architecture_info'] = {'status': 'not_available'}
                
                # Recursive weights info
                if self.recursive_weights_registry:
                    try:
                        registry_stats = self.recursive_weights_registry.get_registry_stats()
                        introspection['recursive_weights_info'] = {
                            'total_weights': registry_stats.get('total_weights', 0),
                            'active_weights': registry_stats.get('active_weights', 0),
                            'cache_hit_rate': registry_stats.get('cache_hit_rate', 0),
                            'stability_rate': registry_stats.get('stability_rate', 0)
                        }
                    except:
                        introspection['recursive_weights_info'] = {'status': 'error_accessing_stats'}
                else:
                    introspection['recursive_weights_info'] = {'status': 'not_initialized'}
                
                # Tool synthesis info
                if self.tool_output_head:
                    try:
                        tool_stats = {
                            'initialized': True,
                            'synthesis_enabled': True,
                            'control_enabled': True
                        }
                        if hasattr(self.tool_output_head, 'get_synthesis_stats'):
                            tool_stats.update(self.tool_output_head.get_synthesis_stats())
                        introspection['tool_synthesis_info'] = tool_stats
                    except:
                        introspection['tool_synthesis_info'] = {'initialized': True, 'stats_unavailable': True}
                else:
                    introspection['tool_synthesis_info'] = {'initialized': False}
            
            # Create introspection display
            tree = Tree("🔍 GPT-Ø System Introspection")
            
            # Model info
            model_node = tree.add("🤖 Model Architecture")
            model_info = introspection.get('model_info', {})
            model_node.add(f"Vocabulary Size: {model_info.get('vocab_size', 'unknown'):,}")
            model_node.add(f"Model Dimension: {model_info.get('d_model', 'unknown')}")
            model_node.add(f"Layers: {model_info.get('n_layers', 'unknown')}")
            model_node.add(f"Attention Heads: {model_info.get('n_heads', 'unknown')}")
            model_node.add(f"Device: {model_info.get('device', 'unknown')}")
            
            # Memory info
            memory_node = tree.add("💾 Neural Memory Runtime")
            memory_info = introspection.get('memory_info', {})
            if memory_info.get('status') == 'not_initialized':
                memory_node.add("Status: Not Initialized")
            else:
                memory_node.add(f"Memory Usage: {memory_info.get('tensor_memory_usage', 0):.1f} MB")
                memory_node.add(f"Memory Capacity: {memory_info.get('tensor_memory_capacity', 0):.1f} MB")
                memory_node.add(f"Utilization: {memory_info.get('memory_utilization_percent', 0):.1f}%")
                memory_node.add(f"Average Importance: {memory_info.get('average_memory_importance', 0):.3f}")
                memory_node.add(f"Active Chunks: {memory_info.get('active_chunks', 0)}")
            
            # Reasoning info
            reasoning_node = tree.add("🧠 Chain-of-Thought Reasoning")
            reasoning_info = introspection.get('reasoning_info', {})
            reasoning_node.add(f"Enabled: {reasoning_info.get('enabled', False)}")
            reasoning_node.add(f"Active Chains: {reasoning_info.get('active_chains_count', 0)}")
            reasoning_node.add(f"Completed Chains: {reasoning_info.get('completed_chains_count', 0)}")
            reasoning_node.add(f"Total Steps: {reasoning_info.get('total_reasoning_steps', 0)}")
            reasoning_node.add(f"Stability Violations: {reasoning_info.get('stability_violations', 0)}")
            
            # Architecture info
            arch_node = tree.add("🏗️ Architecture Details")
            arch_info = introspection.get('architecture_info', {})
            if arch_info.get('status') == 'not_available':
                arch_node.add("Status: Not Available")
            else:
                arch_node.add(f"Total Parameters: {arch_info.get('total_parameters', 0):,}")
                arch_node.add(f"Layers: {arch_info.get('n_layers', 'unknown')}")
                arch_node.add(f"Attention Heads: {arch_info.get('n_heads', 'unknown')}")
                arch_node.add(f"Model Dimension: {arch_info.get('d_model', 'unknown')}")
            
            # Recursive weights info
            rw_node = tree.add("♻️ Recursive Weights System")
            rw_info = introspection.get('recursive_weights_info', {})
            if rw_info.get('status') == 'not_initialized':
                rw_node.add("Status: Not Initialized")
            else:
                rw_node.add(f"Total Weights: {rw_info.get('total_weights', 0)}")
                rw_node.add(f"Active Weights: {rw_info.get('active_weights', 0)}")
                rw_node.add(f"Cache Hit Rate: {rw_info.get('cache_hit_rate', 0):.1%}")
                rw_node.add(f"Stability Rate: {rw_info.get('stability_rate', 0):.1%}")
            
            # Tool synthesis info
            tool_node = tree.add("🛠️ Tool Synthesis System")
            tool_info = introspection.get('tool_synthesis_info', {})
            tool_node.add(f"Initialized: {tool_info.get('initialized', False)}")
            if tool_info.get('initialized'):
                tool_node.add(f"Synthesis Enabled: {tool_info.get('synthesis_enabled', False)}")
                tool_node.add(f"Control Enabled: {tool_info.get('control_enabled', False)}")
            
            self.console.print(tree)
            
        except Exception as e:
            logger.error(f"Introspection error: {e}", exc_info=True)
            self.console.print(f"[red]❌ Introspection failed: {e}[/red]")

    def _debug_memory(self) -> None:
        """Debug memory usage and patterns."""
        if not self.gpt_model:
            self.console.print("[red]❌ Model not initialized[/red]")
            return
        
        try:
            # System memory
            memory = psutil.virtual_memory()
            self.console.print(f"[bold]System Memory:[/bold]")
            self.console.print(f"  Total: {memory.total / 1e9:.1f} GB")
            self.console.print(f"  Available: {memory.available / 1e9:.1f} GB")
            self.console.print(f"  Used: {memory.percent:.1f}%")
            
            # PyTorch memory
            if torch.cuda.is_available():
                self.console.print(f"\n[bold]GPU Memory:[/bold]")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    self.console.print(f"  GPU {i}: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
            
            # Model memory stats
            if hasattr(self.gpt_model, 'get_context_memory_stats'):
                context_stats = self.gpt_model.get_context_memory_stats()
                self.console.print(f"\n[bold]Context Memory:[/bold]")
                self.console.print(f"  Total Tokens: {context_stats.get('total_context_tokens', 0):,}")
                self.console.print(f"  Active Chunks: {context_stats.get('active_chunks', 0)}")
                self.console.print(f"  Capacity Utilization: {context_stats.get('capacity_utilization', 0):.1%}")
            
        except Exception as e:
            logger.error(f"Memory debug error: {e}", exc_info=True)
            self.console.print(f"[red]❌ Memory debug failed: {e}[/red]")

    def _debug_reasoning(self) -> None:
        """Debug reasoning chain state."""
        if not self.gpt_model:
            self.console.print("[red]❌ Model not initialized[/red]")
            return
        
        try:
            cot_metrics = self.gpt_model.get_cot_performance_metrics()
            active_chains = self.gpt_model.get_active_reasoning_chains()
            
            self.console.print(f"[bold]Chain of Thought Performance:[/bold]")
            self.console.print(f"  Total Chains Processed: {cot_metrics.get('total_chains_processed', 0)}")
            self.console.print(f"  Average Chain Length: {cot_metrics.get('average_chain_length', 0):.2f}")
            self.console.print(f"  Stability Violations: {cot_metrics.get('stability_violations', 0)}")
            self.console.print(f"  Contradictions Resolved: {cot_metrics.get('contradictions_resolved', 0)}")
            self.console.print(f"  Ethical Interventions: {cot_metrics.get('ethical_interventions', 0)}")
            
            if active_chains:
                self.console.print(f"\n[bold]Active Reasoning Chains ({len(active_chains)}):[/bold]")
                for chain_id in active_chains[:5]:  # Show first 5
                    chain_state = self.gpt_model.get_chain_state(chain_id)
                    if chain_state:
                        self.console.print(f"  {chain_id}: {chain_state['total_steps']} steps, "
                                         f"{chain_state['overall_stability']} stability")
            
        except Exception as e:
            logger.error(f"Reasoning debug error: {e}", exc_info=True)
            self.console.print(f"[red]❌ Reasoning debug failed: {e}[/red]")

    def _debug_config(self) -> None:
        """Debug configuration state."""
        self.console.print(f"[bold]Configuration State:[/bold]")
        self.console.print(f"  Path: {self.config_state.config_path}")
        self.console.print(f"  Valid: {self.config_state.config_valid}")
        self.console.print(f"  Last Modified: {time.ctime(self.config_state.last_modified)}")
        
        if self.config_state.validation_errors:
            self.console.print(f"\n[bold red]Validation Errors:[/bold red]")
            for error in self.config_state.validation_errors:
                self.console.print(f"  ❌ {error}")
        
        if self.config_state.config_data:
            self.console.print(f"\n[bold]Configuration Sections:[/bold]")
            for section in self.config_state.config_data.keys():
                self.console.print(f"  ✓ {section}")

    def _debug_logs(self) -> None:
        """Show recent log entries."""
        log_file = self.project_root / "gpt_zero_system.log"
        if not log_file.exists():
            self.console.print("[yellow]⚠ Log file not found[/yellow]")
            return
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Show last 20 lines
            recent_lines = lines[-20:] if len(lines) > 20 else lines
            
            self.console.print(f"[bold]Recent Log Entries ({len(recent_lines)} lines):[/bold]")
            for line in recent_lines:
                # Color code by log level
                if "ERROR" in line:
                    self.console.print(f"[red]{line.strip()}[/red]")
                elif "WARNING" in line:
                    self.console.print(f"[yellow]{line.strip()}[/yellow]")
                elif "INFO" in line:
                    self.console.print(f"[blue]{line.strip()}[/blue]")
                else:
                    self.console.print(line.strip())
        
        except Exception as e:
            self.console.print(f"[red]❌ Failed to read logs: {e}[/red]")

    def _debug_torch(self) -> None:
        """Debug PyTorch state and configuration."""
        self.console.print(f"[bold]PyTorch Configuration:[/bold]")
        self.console.print(f"  Version: {torch.__version__}")
        self.console.print(f"  CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.console.print(f"  CUDA Version: {torch.version.cuda}")
            self.console.print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            self.console.print(f"  Device Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.console.print(f"  Device {i}: {props.name}")
                self.console.print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
                self.console.print(f"    Compute Capability: {props.major}.{props.minor}")

    def _handle_monitor_mode(self) -> None:
        """Handle real-time system monitoring."""
        self.console.print("\n[bold blue]📊 System Monitor[/bold blue]")
        self.console.print("Press Ctrl+C to return to main menu\n")
        
        try:
            with Live(self._create_status_display(), refresh_per_second=2) as live:
                while not self.shutdown_requested:
                    try:
                        live.update(self._create_status_display())
                        time.sleep(0.5)
                    except KeyboardInterrupt:
                        break
        except Exception as e:
            logger.error(f"Monitor mode error: {e}", exc_info=True)
            self.console.print(f"[red]❌ Monitor error: {e}[/red]")

    def _handle_tools_mode(self) -> None:
        """Handle tool output and synthesis mode."""
        if not self.gpt_model or not self.tool_output_head:
            self.console.print("[red]❌ Model or tool head not initialized[/red]")
            return
        
        self.console.print("\n[bold magenta]🛠️ Tool Synthesis & Control Mode[/bold magenta]")
        self.console.print("Generate autonomous tool outputs and examine synthesis capabilities")
        self.console.print("Commands: 'synthesize <objective>', 'control <system>', 'back'\n")
        
        while not self.shutdown_requested:
            try:
                prompt = Prompt.ask("Tool command", default="back")
                
                if prompt.lower() in ['back', 'exit']:
                    break
                
                # Parse command
                parts = prompt.split(' ', 1)
                command = parts[0].lower()
                objective = parts[1] if len(parts) > 1 else ""
                
                if command == "synthesize":
                    if not objective:
                        objective = Prompt.ask("Tool synthesis objective")
                    
                    try:
                        with self.console.status("[bold magenta]🔧 Synthesizing tools...") as status:
                            # Encode objective
                            input_tokens = self.tokenizer.encode(objective)
                            if not isinstance(input_tokens, torch.Tensor):
                                input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
                            
                            # Generate tool output using the universal tool head
                            tool_result = self.tool_output_head.synthesize_tool_response(
                                context_tokens=input_tokens,
                                objective=objective,
                                safety_constraints={"max_risk_level": "medium"},
                                enable_autonomous_discovery=True
                            )
                            
                            # Display results
                            if tool_result:
                                self.console.print(f"\n[bold green]🎯 Tool Synthesis Complete:[/bold green]")
                                
                                # Create results table
                                results_table = Table(title="Synthesized Tools", show_header=True, header_style="bold blue")
                                results_table.add_column("Component", style="cyan")
                                results_table.add_column("Result", style="green")
                                results_table.add_column("Details", style="yellow")
                                
                                for key, value in tool_result.items():
                                    if isinstance(value, dict):
                                        details = ", ".join([f"{k}: {v}" for k, v in value.items()][:3])
                                        results_table.add_row(key, "Generated", details)
                                    elif isinstance(value, list):
                                        results_table.add_row(key, f"List ({len(value)} items)", str(value[:2]))
                                    else:
                                        results_table.add_row(key, "Generated", str(value)[:50])
                                
                                self.console.print(results_table)
                            else:
                                self.console.print("[yellow]⚠ No tool output generated[/yellow]")
                                
                    except Exception as e:
                        logger.error(f"Tool synthesis error: {e}", exc_info=True)
                        self.console.print(f"[red]❌ Tool synthesis error: {e}[/red]")
                
                elif command == "control":
                    if not objective:
                        objective = Prompt.ask("System control objective")
                    
                    try:
                        with self.console.status("[bold magenta]🎛️ Generating system control...") as status:
                            # Encode control objective
                            input_tokens = self.tokenizer.encode(objective)
                            if not isinstance(input_tokens, torch.Tensor):
                                input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
                            
                            # Generate control commands
                            control_result = self.tool_output_head.generate_system_control(
                                context_tokens=input_tokens,
                                control_objective=objective,
                                safety_validation=True
                            )
                            
                            # Display control results
                            if control_result:
                                self.console.print(f"\n[bold blue]🎛️ System Control Generated:[/bold blue]")
                                
                                # Display control commands
                                for category, commands in control_result.items():
                                    if isinstance(commands, list):
                                        self.console.print(f"\n[bold]{category.title()}:[/bold]")
                                        for i, cmd in enumerate(commands[:5], 1):  # Show first 5
                                            self.console.print(f"  {i}. {cmd}")
                                    else:
                                        self.console.print(f"{category}: {commands}")
                            else:
                                self.console.print("[yellow]⚠ No control commands generated[/yellow]")
                                
                    except Exception as e:
                        logger.error(f"System control error: {e}", exc_info=True)
                        self.console.print(f"[red]❌ System control error: {e}[/red]")
                
                elif command == "status":
                    # Show tool head status
                    if hasattr(self.tool_output_head, 'get_status'):
                        status_info = self.tool_output_head.get_status()
                        self.console.print(f"[bold cyan]Tool Head Status:[/bold cyan]")
                        for key, value in status_info.items():
                            self.console.print(f"  {key}: {value}")
                    else:
                        self.console.print("[yellow]Tool head status not available[/yellow]")
                
                else:
                    self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
                    self.console.print("Available commands: synthesize, control, status, back")
            
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Tools mode interrupted[/yellow]")
                break
            except Exception as e:
                logger.error(f"Tools mode error: {e}", exc_info=True)
                self.console.print(f"[red]❌ Tools error: {e}[/red]")
                self._record_error("tools", str(e))

    def _record_error(self, error_type: str, error_message: str) -> None:
        """Record error for tracking and analysis."""
        error_record = {
            'timestamp': time.time(),
            'type': error_type,
            'message': error_message,
            'correlation_id': getattr(threading.current_thread(), 'correlation_id', 'UNKNOWN')
        }
        
        self.error_history.append(error_record)
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

    def _cleanup_resources(self) -> None:
        """Clean up system resources during shutdown."""
        try:
            logger.info("Starting resource cleanup...")
            
            # Stop monitoring
            if self.monitor_thread and self.monitor_thread.is_alive():
                logger.info("Stopping monitoring thread...")
                self.shutdown_event.set()
                self.monitor_thread.join(timeout=5.0)
            
            # Clean up model resources
            if self.gpt_model:
                logger.info("Cleaning up model resources...")
                # Save any important state
                try:
                    state_file = self.project_root / "gpt_zero_final_state.pth"
                    self.gpt_model.save_state(str(state_file))
                    logger.info(f"Model state saved to {state_file}")
                except Exception as e:
                    logger.error(f"Failed to save model state: {e}")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)

    def run(self) -> int:
        """
        Main execution loop with comprehensive error handling.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Display startup banner
            self._display_startup_banner()
            
            # Validate environment
            env_errors = self._validate_environment()
            if env_errors:
                self.console.print("[bold red]❌ Environment Validation Failed:[/bold red]")
                for error in env_errors:
                    self.console.print(f"  • {error}")
                
                if not Confirm.ask("Continue anyway?", default=False):
                    return 1
            
            # Load and validate configuration
            if not self._load_and_validate_config():
                self.console.print("[bold red]❌ Configuration validation failed[/bold red]")
                for error in self.config_state.validation_errors:
                    self.console.print(f"  • {error}")
                return 1
            
            # Initialize system components
            if not self._initialize_system_components():
                self.console.print("[bold red]❌ System initialization failed[/bold red]")
                return 1
            
            # Main interaction loop
            self.system_state = SystemState.RUNNING
            self._run_main_loop()
            
            return 0
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            logger.critical(f"Fatal error: {e}", exc_info=True)
            self.console.print(f"[bold red]FATAL ERROR: {e}[/bold red]")
            return 1
        finally:
            self.system_state = SystemState.SHUTTING_DOWN
            self._cleanup_resources()
            self.system_state = SystemState.SHUTDOWN
            self.console.print("[bold green]GPT-0 System Shutdown Complete[/bold green]")

    def _display_startup_banner(self) -> None:
        """Display system startup banner."""
        banner = """
        ================================================================
                                GPT-0 System                         
                        Self-Modifying AI Platform                 
                                                               
          Multimodal Intelligence      Tool Synthesis             
          Self-Modification           Real-time Monitoring        
          Chain of Thought            Safety & Ethics            
                                                               
                        Author: Morpheus                          
                       Version: 2.0.0 (Production)               
        ================================================================
        """
        
        self.console.print(banner, style="bold cyan")
        self.console.print(f"[dim]Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

    def _run_main_loop(self) -> None:
        """Run the main interaction loop."""
        
        async def main_async():
            mode_handlers = {
                'chat': self._handle_chat_mode,
                'debug': self._handle_debug_mode,
                'monitor': self._handle_monitor_mode,
                'tools': self._handle_tools_mode,
                'config': self._handle_config_mode,
                'quit': lambda: setattr(self, 'shutdown_requested', True)
            }
            
            while not self.shutdown_requested:
                try:
                    # Display current status
                    self.console.print(f"\n[bold]System Status:[/bold] {self.system_state.value}")
                    self.console.print(f"[bold]Current Mode:[/bold] {self.interaction_mode.value}")
                    
                    # Show available commands
                    self.console.print("\n[bold]Available Commands:[/bold]")
                    for cmd in mode_handlers.keys():
                        self.console.print(f"  [cyan]{cmd}[/cyan]")
                    
                    # Get user input
                    command = Prompt.ask(
                        "Command",
                        choices=list(mode_handlers.keys()),
                        default="chat"
                    )
                    
                    handler = mode_handlers.get(command)
                    if handler:
                        if asyncio.iscoroutinefunction(handler):
                            await handler()
                        else:
                            handler()
                    
                except (KeyboardInterrupt, EOFError):
                    self.shutdown_requested = True
                    break
                except Exception as e:
                    logger.error(f"Main loop error: {e}", exc_info=True)
                    self.console.print(f"[red]❌ Error: {e}[/red]")

        asyncio.run(main_async())

    def _handle_config_mode(self) -> None:
        """Handle configuration management mode."""
        self.console.print("\n[bold cyan]⚙️ Configuration Management[/bold cyan]")
        
        config_commands = {
            'show': self._config_show,
            'validate': self._config_validate,
            'reload': self._config_reload,
            'edit': self._config_edit,
            'back': lambda: None
        }
        
        while not self.shutdown_requested:
            self.console.print("\nConfiguration Commands:")
            for cmd in config_commands.keys():
                self.console.print(f"  [cyan]{cmd}[/cyan]")
            
            command = Prompt.ask("Config command", choices=list(config_commands.keys()), default="back")
            
            if command == 'back':
                break
            
            try:
                config_commands[command]()
            except Exception as e:
                logger.error(f"Config command error: {e}", exc_info=True)
                self.console.print(f"[red]=> Config error: {e}[/red]")

    def _config_show(self) -> None:
        """Show current configuration."""
        if self.config_state.config_data:
            syntax = Syntax(
                json.dumps(self.config_state.config_data, indent=2),
                "json",
                theme="monokai",
                line_numbers=True
            )
            self.console.print(Panel(syntax, title="Current Configuration"))
        else:
            self.console.print("[yellow]⚠ No configuration loaded[/yellow]")

    def _config_validate(self) -> None:
        """Validate current configuration."""
        if self._load_and_validate_config():
            self.console.print("[green]✓ Configuration is valid[/green]")
        else:
            self.console.print("[red]❌ Configuration validation failed[/red]")
            for error in self.config_state.validation_errors:
                self.console.print(f"  • {error}")

    def _config_reload(self) -> None:
        """Reload configuration from file."""
        if self._load_and_validate_config():
            self.console.print("[green]✓ Configuration reloaded successfully[/green]")
        else:
            self.console.print("[red]❌ Configuration reload failed[/red]")

    def _config_edit(self) -> None:
        """Open configuration file for editing."""
        import subprocess
        import shutil
        
        # Find a suitable editor
        editors = ['code', 'notepad', 'nano', 'vim']
        editor = None
        
        for ed in editors:
            if shutil.which(ed):
                editor = ed
                break
        
        if editor:
            try:
                subprocess.run([editor, str(self.config_path)])
                self.console.print(f"[green]✓ Opened {self.config_path} with {editor}[/green]")
                
                if Confirm.ask("Reload configuration?", default=True):
                    self._config_reload()
            except Exception as e:
                self.console.print(f"[red]=> Failed to open editor: {e}[/red]")
        else:
            self.console.print(f"[yellow]⚠ No suitable editor found. Edit {self.config_path} manually.[/yellow]")


def main():
    """Main entry point for the GPT-Ø system launcher."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GPT-Ø System Launcher - Self-Modifying Multimodal AI Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                           # Use default config
  python run.py --config custom.json     # Use custom config
  python run.py --debug                  # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config/agent_config.json)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='GPT-Ø System v2.0.0 (Production)'
    )
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Create and run the launcher
        launcher = GPTZeroLauncher(config_path=args.config)
        exit_code = launcher.run()
        sys.exit(exit_code)
        
    except Exception as e:
        console = Console()
        console.print(f"[bold red]FATAL STARTUP ERROR: {e}[/bold red]")
        logger.critical(f"Fatal startup error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# =============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL INTEGRATION
# =============================================================================

def create_gpt_zero_system(config_path: Optional[str] = None) -> Optional[GPTZeroLauncher]:
    """
    Create and initialize a GPT-Ø system for external use (e.g., web server integration).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized GPTZeroLauncher instance or None if initialization fails
    """
    try:
        # Suppress startup banner for external usage
        launcher = GPTZeroLauncher(config_path=config_path)
        
        # Validate environment
        env_errors = launcher._validate_environment()
        if env_errors:
            logger.error(f"Environment validation failed: {env_errors}")
            return None
        
        # Load and validate configuration
        if not launcher._load_and_validate_config():
            logger.error(f"Configuration validation failed: {launcher.config_state.validation_errors}")
            return None
        
        # Initialize system components
        if not launcher._initialize_system_components():
            logger.error("Failed to initialize system components")
            return None
        
        logger.info("GPT-Ø system created successfully for external integration")
        return launcher
        
    except Exception as e:
        logger.error(f"Failed to create GPT-Ø system: {e}", exc_info=True)
        return None

def get_system_components(launcher: GPTZeroLauncher) -> Dict[str, Any]:
    """
    Extract system components from an initialized launcher.
    
    Args:
        launcher: Initialized GPTZeroLauncher instance
        
    Returns:
        Dictionary containing system components
    """
    return {
        'gpt_model': launcher.gpt_model,
        'tokenizer': launcher.tokenizer,
        'config_orchestrator': launcher.config_orchestrator,
        'device': launcher.device,
        'config_state': launcher.config_state,
        'metrics': launcher.metrics
    }

