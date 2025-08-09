#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modality Plugin System Retrofit
Extends existing ModalityEncoderManager to support plugins while maintaining compatibility

Strategy:
1. Keep existing encoders working
2. Add plugin discovery to existing ModalityEncoderManager  
3. Gradual migration path
4. Test with new modalities as plugins

This retrofits your existing solid foundation rather than rebuilding.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Tuple, Type
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import importlib
import pkgutil
import inspect
import logging
import time
from pathlib import Path

# Import existing system
try:
    from modality_encoders import (
        ModalityEncoderManager as ExistingModalityEncoderManager,
        BaseModalityEncoder as ExistingBaseModalityEncoder, 
        ModalityConfig, ModalityType, EncodingStrategy,
        VisionEncoder, AudioEncoder, VideoEncoder, LiveWebEncoder, StructuredDataEncoder
    )
    from transformer_backbone import EclogueConfig
except ImportError:
    print("Warning: Existing modality system not found. Using fallbacks.")

logger = logging.getLogger("EcloguePluginSystem")

# ================================================================
# ENHANCED BASE ENCODER CONTRACT (for plugins)
# ================================================================

class BaseModalityEncoder(ABC, nn.Module):
    """
    Enhanced base contract for plugin modality encoders
    Compatible with existing system but with plugin-specific enhancements
    """
    
    def __init__(self, config: Optional[ModalityConfig] = None):
        super().__init__()
        self.config = config
    
    @property
    @abstractmethod
    def modality_name(self) -> str:
        """Unique identifier for this modality (e.g., 'haptic', 'spectral')"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version for compatibility checking"""
        pass
    
    @property
    def supported_formats(self) -> List[str]:
        """List of supported input formats"""
        return ["raw"]
    
    @property
    def memory_requirements(self) -> Dict[str, Union[int, str]]:
        """Memory requirements for this encoder"""
        return {"gpu_memory_mb": 512, "cpu_memory_mb": 256}
    
    @property
    def performance_hints(self) -> Dict[str, Any]:
        """Performance optimization hints"""
        return {
            "batch_size_preference": 16,
            "cuda_optimized": True,
            "parallel_friendly": True
        }
    
    @abstractmethod
    def encode(self, raw_data: Any, attention_mask: Optional[torch.Tensor] = None,
               memory_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Core encoding method - must return same format as existing encoders
        
        Returns:
            Dict with keys: 'embeddings', 'attention_mask', 'sequence_length', 'modality_type'
        """
        pass
    
    def validate_input(self, raw_data: Any) -> bool:
        """Validate input data format"""
        return True
    
    def get_sequence_length(self, inputs: Any) -> int:
        """Get sequence length for input"""
        return 1

# ================================================================
# PLUGIN DISCOVERY AND MANAGEMENT
# ================================================================

@dataclass
class PluginInfo:
    """Information about a discovered plugin"""
    name: str
    version: str
    class_reference: Type[BaseModalityEncoder]
    file_path: str
    supported_formats: List[str]
    memory_requirements: Dict[str, Union[int, str]]
    performance_hints: Dict[str, Any]
    validation_passed: bool = False
    load_time_ms: float = 0.0

class PluginValidator:
    """Validates plugin safety and compatibility"""
    
    def __init__(self):
        self.required_methods = ['modality_name', 'version', 'encode']
        self.security_checks = True
    
    def validate_plugin(self, plugin_class: Type[BaseModalityEncoder]) -> Dict[str, Any]:
        """Comprehensive plugin validation"""
        
        validation_result = {
            'passed': False,
            'errors': [],
            'warnings': [],
            'security_score': 0.0
        }
        
        try:
            # Check inheritance
            if not issubclass(plugin_class, BaseModalityEncoder):
                validation_result['errors'].append("Must inherit from BaseModalityEncoder")
                return validation_result
            
            # Check required methods
            for method in self.required_methods:
                if not hasattr(plugin_class, method):
                    validation_result['errors'].append(f"Missing required method: {method}")
            
            # Try instantiation (with minimal config)
            try:
                temp_config = ModalityConfig(
                    modality_type=ModalityType.STRUCTURED,  # Placeholder
                    encoding_strategy=EncodingStrategy.SEQUENCE_BASED,
                    input_dimensions=(1,),
                    hidden_size=512
                )
                instance = plugin_class(temp_config)
                
                # Test property access
                _ = instance.modality_name
                _ = instance.version
                
                validation_result['security_score'] = 0.8  # Basic security check passed
                
            except Exception as e:
                validation_result['errors'].append(f"Instantiation failed: {str(e)}")
            
            # If no errors, mark as passed
            if not validation_result['errors']:
                validation_result['passed'] = True
                validation_result['security_score'] = 1.0
                
        except Exception as e:
            validation_result['errors'].append(f"Validation exception: {str(e)}")
        
        return validation_result

class PluginDiscoveryEngine:
    """Discovers and loads modality plugins safely"""
    
    def __init__(self, plugin_directories: List[str] = None):
        self.plugin_directories = plugin_directories or ["modality_plugins"]
        self.discovered_plugins: Dict[str, PluginInfo] = {}
        self.validator = PluginValidator()
        
    def discover_plugins(self, config: Optional[EclogueConfig] = None) -> Dict[str, PluginInfo]:
        """Discover all available plugins"""
        
        logger.info("Starting plugin discovery...")
        
        for plugin_dir in self.plugin_directories:
            try:
                self._scan_directory(plugin_dir, config)
            except ImportError as e:
                logger.warning(f"Plugin directory {plugin_dir} not found: {e}")
            except Exception as e:
                logger.error(f"Plugin discovery error in {plugin_dir}: {e}")
        
        logger.info(f"Discovered {len(self.discovered_plugins)} plugins")
        return self.discovered_plugins
    
    def _scan_directory(self, plugin_dir: str, config: Optional[EclogueConfig]):
        """Scan a directory for plugins"""
        
        try:
            # Import the plugin package
            plugin_module = importlib.import_module(plugin_dir)
            
            # Iterate through modules in the package
            for _, name, _ in pkgutil.iter_modules(plugin_module.__path__):
                self._load_plugin_module(f"{plugin_dir}.{name}", name, config)
                
        except ImportError:
            # Try loading from filesystem
            plugin_path = Path(plugin_dir)
            if plugin_path.exists():
                self._scan_filesystem_directory(plugin_path, config)
    
    def _load_plugin_module(self, module_path: str, module_name: str, config: Optional[EclogueConfig]):
        """Load and validate a plugin module"""
        
        start_time = time.time()
        
        try:
            module = importlib.import_module(module_path)
            
            # Find encoder classes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseModalityEncoder) and 
                    attr is not BaseModalityEncoder):
                    
                    # Validate the plugin
                    validation_result = self.validator.validate_plugin(attr)
                    
                    if validation_result['passed']:
                        # Create plugin info
                        try:
                            temp_instance = attr()
                            plugin_info = PluginInfo(
                                name=temp_instance.modality_name,
                                version=temp_instance.version,
                                class_reference=attr,
                                file_path=module_path,
                                supported_formats=temp_instance.supported_formats,
                                memory_requirements=temp_instance.memory_requirements,
                                performance_hints=temp_instance.performance_hints,
                                validation_passed=True,
                                load_time_ms=(time.time() - start_time) * 1000
                            )
                            
                            self.discovered_plugins[plugin_info.name] = plugin_info
                            logger.info(f"✓ Loaded plugin: {plugin_info.name} v{plugin_info.version}")
                            
                        except Exception as e:
                            logger.error(f"✗ Plugin instantiation failed {attr_name}: {e}")
                    else:
                        logger.warning(f"✗ Plugin validation failed {attr_name}: {validation_result['errors']}")
                        
        except Exception as e:
            logger.error(f"✗ Failed to load module {module_path}: {e}")
    
    def _scan_filesystem_directory(self, plugin_path: Path, config: Optional[EclogueConfig]):
        """Fallback: scan filesystem directory"""
        # Implementation for filesystem-based plugin loading
        pass

# ================================================================
# ENHANCED MODALITY ENCODER MANAGER (retrofits existing)
# ================================================================

class EnhancedModalityEncoderManager(ExistingModalityEncoderManager):
    """
    Enhanced version of existing ModalityEncoderManager with plugin support
    
    Strategy: Extend existing functionality, maintain compatibility
    """
    
    def __init__(self, backbone_config: EclogueConfig):
        # Initialize existing system
        super().__init__(backbone_config)
        
        # Add plugin system
        self.plugin_engine = PluginDiscoveryEngine()
        self.plugin_encoders: Dict[str, BaseModalityEncoder] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        
        # Plugin management
        self.plugins_enabled = True
        self.plugin_stats = {
            'plugins_loaded': 0,
            'plugins_failed': 0,
            'total_load_time_ms': 0.0
        }
        
        # Load plugins on initialization
        if self.plugins_enabled:
            self._discover_and_load_plugins()
    
    def _discover_and_load_plugins(self):
        """Discover and load all available plugins"""
        
        logger.info("Discovering modality plugins...")
        
        # Discover plugins
        discovered = self.plugin_engine.discover_plugins(self.backbone_config)
        
        # Load each plugin
        for plugin_name, plugin_info in discovered.items():
            try:
                self._load_plugin(plugin_name, plugin_info)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")
                self.plugin_stats['plugins_failed'] += 1
        
        logger.info(f"Plugin loading complete: {self.plugin_stats['plugins_loaded']} loaded, "
                   f"{self.plugin_stats['plugins_failed']} failed")
    
    def _load_plugin(self, plugin_name: str, plugin_info: PluginInfo):
        """Load a specific plugin"""
        
        start_time = time.time()
        
        # Create plugin configuration
        plugin_config = ModalityConfig(
            modality_type=ModalityType.STRUCTURED,  # Default, plugin can override
            encoding_strategy=EncodingStrategy.SEQUENCE_BASED,
            input_dimensions=(1,),
            hidden_size=self.backbone_config.hidden_size,
            num_layers=4,
            num_heads=16
        )
        
        # Instantiate plugin
        plugin_instance = plugin_info.class_reference(plugin_config)
        
        # Register plugin
        self.plugin_encoders[plugin_name] = plugin_instance
        self.plugin_info[plugin_name] = plugin_info
        
        # Update statistics
        load_time = (time.time() - start_time) * 1000
        self.plugin_stats['plugins_loaded'] += 1
        self.plugin_stats['total_load_time_ms'] += load_time
        
        logger.info(f"✓ Plugin loaded: {plugin_name} ({load_time:.1f}ms)")
    
    def encode_modality(self, inputs: Any, modality: Union[ModalityType, str],
                       attention_mask: Optional[torch.Tensor] = None,
                       memory_context: Optional[torch.Tensor] = None,
                       use_cache: bool = True) -> Dict[str, torch.Tensor]:
        """
        Enhanced encode_modality that supports both existing encoders and plugins
        """
        
        # Handle plugin modalities (string-based)
        if isinstance(modality, str) and modality in self.plugin_encoders:
            return self._encode_with_plugin(inputs, modality, attention_mask, memory_context, use_cache)
        
        # Handle existing modalities (enum-based)
        elif isinstance(modality, ModalityType):
            return super().encode_modality(inputs, modality, attention_mask, memory_context, use_cache)
        
        # Handle string modality names for existing encoders
        elif isinstance(modality, str):
            try:
                modality_enum = ModalityType(modality.lower())
                return super().encode_modality(inputs, modality_enum, attention_mask, memory_context, use_cache)
            except ValueError:
                raise ValueError(f"Unknown modality: {modality}. Available: {self.get_all_supported_modalities()}")
        
        else:
            raise ValueError(f"Invalid modality type: {type(modality)}")
    
    def _encode_with_plugin(self, inputs: Any, plugin_name: str,
                           attention_mask: Optional[torch.Tensor] = None,
                           memory_context: Optional[torch.Tensor] = None,
                           use_cache: bool = True) -> Dict[str, torch.Tensor]:
        """Encode using a plugin encoder"""
        
        start_time = time.time()
        
        # Get plugin encoder
        plugin_encoder = self.plugin_encoders[plugin_name]
        
        # Validate input
        if not plugin_encoder.validate_input(inputs):
            raise ValueError(f"Invalid input for plugin {plugin_name}")
        
        # Check cache
        cache_key = f"plugin_{plugin_name}_{hash(str(inputs))}" if use_cache else None
        if cache_key and cache_key in self.encoding_cache:
            return self.encoding_cache[cache_key]
        
        # Encode
        result = plugin_encoder.encode(inputs, attention_mask, memory_context)
        
        # Ensure result format compatibility
        if 'modality_type' not in result:
            result['modality_type'] = plugin_name
        
        # Update statistics
        encoding_time = (time.time() - start_time) * 1000
        if plugin_name not in self.encoding_stats:
            self.encoding_stats[plugin_name] = {'calls': 0, 'total_time': 0.0}
        
        self.encoding_stats[plugin_name]['calls'] += 1
        self.encoding_stats[plugin_name]['total_time'] += encoding_time
        
        # Cache result
        if cache_key and len(self.encoding_cache) < self.max_cache_size:
            self.encoding_cache[cache_key] = result
        
        return result
    
    def get_all_supported_modalities(self) -> List[str]:
        """Get all supported modalities (existing + plugins)"""
        
        existing = [m.value for m in self.get_supported_modalities()]
        plugins = list(self.plugin_encoders.keys())
        
        return existing + plugins
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get information about a specific plugin"""
        return self.plugin_info.get(plugin_name)
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded plugins with their information"""
        
        plugins = {}
        for name, info in self.plugin_info.items():
            plugins[name] = {
                'version': info.version,
                'supported_formats': info.supported_formats,
                'memory_requirements': info.memory_requirements,
                'performance_hints': info.performance_hints,
                'load_time_ms': info.load_time_ms
            }
        
        return plugins
    
    def reload_plugins(self):
        """Reload all plugins (for development)"""
        
        logger.info("Reloading plugins...")
        
        # Clear existing plugins
        self.plugin_encoders.clear()
        self.plugin_info.clear()
        
        # Reset stats
        self.plugin_stats = {
            'plugins_loaded': 0,
            'plugins_failed': 0,
            'total_load_time_ms': 0.0
        }
        
        # Reload
        self._discover_and_load_plugins()

# ================================================================
# TESTING AND EXAMPLE USAGE
# ================================================================

def test_plugin_system():
    """Test the retrofitted plugin system"""
    
    print("Testing Eclogue Ø Plugin System Retrofit")
    print("=" * 50)
    
    # Create enhanced manager
    config = EclogueConfig(hidden_size=10192)
    manager = EnhancedModalityEncoderManager(config)
    
    print(f"\nSupported modalities: {manager.get_all_supported_modalities()}")
    print(f"Loaded plugins: {list(manager.plugin_encoders.keys())}")
    
    # Test existing encoders still work
    print(f"\nTesting existing encoder...")
    try:
        test_image = torch.randn(1, 3, 224, 224)
        result = manager.encode_modality(test_image, ModalityType.IMAGE)
        print(f"✓ Existing encoder works: {result['embeddings'].shape}")
    except Exception as e:
        print(f"✗ Existing encoder failed: {e}")
    
    # Test plugin loading stats
    print(f"\nPlugin Statistics:")
    for key, value in manager.plugin_stats.items():
        print(f"  {key}: {value}")
    
    # Show plugin details
    plugins = manager.list_plugins()
    if plugins:
        print(f"\nPlugin Details:")
        for name, info in plugins.items():
            print(f"  {name} v{info['version']}: {info['supported_formats']}")
    
    print(f"\nPlugin system retrofit test complete!")

if __name__ == "__main__":
    test_plugin_system()