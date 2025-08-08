"""
Cognitive Architecture Specification (CAS) System v1.0
======================================================

Self-contained, reusable system for advanced AI model configuration.
Integrates with existing model_creation.py workflow.

Features:
- CAS YAML parsing and validation
- Constitutional safety framework
- Cognitive profile management  
- Runtime adaptation configuration
- Platform translation (Ollama/GGUF)
- Complete integration hooks

Author: Cybernetic Architecture Division
License: MIT
Dependencies: pyyaml, torch, numpy, psutil
"""

import yaml
import json
import hashlib
import time
import re
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime, timezone
import copy
import threading
from collections import defaultdict, deque
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class SafetyMode(Enum):
    """Constitutional framework safety modes"""
    STRICT = "strict"
    BALANCED = "balanced"
    PERMISSIVE = "permissive"
    DISABLED = "disabled"


class EnforcementLevel(Enum):
    """How to handle constitutional violations"""
    HARD_FAIL = "hard_fail"
    WARN_AND_PROCEED = "warn_and_proceed"
    LOG_ONLY = "log_only"


class CognitiveProfile(Enum):
    """Available cognitive profiles"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"
    RESEARCH = "research"
    TECHNICAL = "technical"
    TEACHING = "teaching"


class ModelType(Enum):
    """Supported model types"""
    GGUF = "gguf"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    BITNET = "bitnet"
    SAFETENSORS = "safetensors"


class ViolationType(Enum):
    """Types of constitutional violations"""
    CONTENT_FILTER = "content_filter"
    SAFETY_PRINCIPLE = "safety_principle"
    CONTEXT_VIOLATION = "context_violation"
    PATTERN_DETECTION = "pattern_detection"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyPattern:
    """Pattern-based safety detection rule"""
    name: str
    pattern: str
    risk_level: RiskLevel
    description: str
    enabled: bool = True
    case_sensitive: bool = False
    regex_flags: int = 0


@dataclass
class ViolationAnalysis:
    """Detailed analysis of a constitutional violation"""
    violation_type: ViolationType
    risk_level: RiskLevel
    confidence: float
    matched_patterns: List[str]
    context_window: str
    suggested_action: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class BaseModelConfig:
    """Base model specification"""
    provider: str = "local"
    path: str = ""
    type: str = "gguf"
    architecture: str = "llama"
    parameter_count: str = "7B"
    context_length: int = 8192
    quantization: Optional[str] = "Q4_K_M"


@dataclass
class ConstitutionalFramework:
    """Constitutional AI safety configuration"""
    governor_mode: str = "balanced"
    enforcement_level: str = "warn_and_proceed"
    safety_principles: Dict[str, List[str]] = field(default_factory=dict)
    content_filters: Dict[str, str] = field(default_factory=dict)
    override_permissions: Dict[str, bool] = field(default_factory=dict)


@dataclass
class CognitiveProfileConfig:
    """Individual cognitive profile configuration"""
    name: str
    description: str
    system_prompt: str
    reasoning_framework: str
    parameter_preferences: Dict[str, float] = field(default_factory=dict)


@dataclass
class ReasoningFramework:
    """Reasoning framework specification"""
    description: str
    steps: List[str]
    validation_criteria: List[str]


@dataclass
class RuntimeAdaptation:
    """Runtime parameter adaptation configuration"""
    enabled: bool = True
    adaptation_frequency: int = 25
    learning_rate_global: float = 0.02
    adaptive_parameters: Dict[str, Any] = field(default_factory=dict)
    adaptation_rules: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryProfile:
    """Memory management configuration"""
    max_memory_gb: float = 6.0
    memory_management: str = "adaptive"
    tier_allocation: Dict[str, float] = field(default_factory=dict)
    caching_policies: List[Dict[str, Any]] = field(default_factory=list)
    compression_config: Dict[str, Any] = field(default_factory=dict)
    optimization_triggers: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PlatformTranslation:
    """Platform compatibility configuration"""
    ollama: Dict[str, Any] = field(default_factory=dict)
    gguf: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CASMetadata:
    """CAS file metadata"""
    system_model_id: str
    custom_name: str
    version: str = "1.0.0"
    author: str = "Unknown"
    description: str = ""
    created_at: str = ""
    cas_version: str = "1.0"
    sha256_hash: str = ""
    project_context: str = ""


@dataclass
class CASSpecification:
    """Complete CAS specification"""
    metadata: CASMetadata
    base_model: BaseModelConfig
    constitutional_framework: ConstitutionalFramework
    cognitive_architecture: Dict[str, Any]
    runtime_adaptation: RuntimeAdaptation
    memory_profile: MemoryProfile
    platform_translation: PlatformTranslation
    development_config: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)


class CASValidator:
    """Validates CAS specifications for correctness and safety"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, cas_spec: CASSpecification) -> Tuple[bool, List[str], List[str]]:
        """Validate complete CAS specification"""
        self.errors = []
        self.warnings = []
        
        # Validate each section
        self._validate_metadata(cas_spec.metadata)
        self._validate_base_model(cas_spec.base_model)
        self._validate_constitutional_framework(cas_spec.constitutional_framework)
        self._validate_cognitive_architecture(cas_spec.cognitive_architecture)
        self._validate_runtime_adaptation(cas_spec.runtime_adaptation)
        self._validate_memory_profile(cas_spec.memory_profile)
        self._validate_platform_translation(cas_spec.platform_translation)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors.copy(), self.warnings.copy()
    
    def _validate_metadata(self, metadata: CASMetadata):
        """Validate metadata section"""
        if not metadata.system_model_id:
            self.errors.append("system_model_id is required")
        
        if not metadata.custom_name:
            self.errors.append("custom_name is required")
        
        if not metadata.cas_version:
            self.errors.append("cas_version is required")
    
    def _validate_base_model(self, base_model: BaseModelConfig):
        """Validate base model configuration"""
        if not base_model.path:
            self.errors.append("base_model.path is required")
        
        valid_types = [t.value for t in ModelType]
        if base_model.type not in valid_types:
            self.errors.append(f"base_model.type must be one of: {valid_types}")
        
        if base_model.context_length <= 0:
            self.errors.append("base_model.context_length must be positive")
    
    def _validate_constitutional_framework(self, framework: ConstitutionalFramework):
        """Validate constitutional framework"""
        valid_modes = [m.value for m in SafetyMode]
        if framework.governor_mode not in valid_modes:
            self.errors.append(f"constitutional_framework.governor_mode must be one of: {valid_modes}")
        
        valid_enforcement = [e.value for e in EnforcementLevel]
        if framework.enforcement_level not in valid_enforcement:
            self.errors.append(f"constitutional_framework.enforcement_level must be one of: {valid_enforcement}")
    
    def _validate_cognitive_architecture(self, architecture: Dict[str, Any]):
        """Validate cognitive architecture"""
        if "default_profile" not in architecture:
            self.errors.append("cognitive_architecture.default_profile is required")
        
        if "profiles" not in architecture:
            self.errors.append("cognitive_architecture.profiles is required")
        else:
            default_profile = architecture.get("default_profile")
            profiles = architecture.get("profiles", {})
            
            if default_profile and default_profile not in profiles:
                self.errors.append(f"default_profile '{default_profile}' not found in profiles")
            
            # Validate each profile
            for profile_name, profile_config in profiles.items():
                if not isinstance(profile_config, dict):
                    self.errors.append(f"Profile '{profile_name}' must be a dictionary")
                    continue
                
                required_fields = ["name", "description", "system_prompt", "reasoning_framework"]
                for field in required_fields:
                    if field not in profile_config:
                        self.errors.append(f"Profile '{profile_name}' missing required field: {field}")
    
    def _validate_runtime_adaptation(self, adaptation: RuntimeAdaptation):
        """Validate runtime adaptation configuration"""
        if adaptation.adaptation_frequency <= 0:
            self.errors.append("runtime_adaptation.adaptation_frequency must be positive")
        
        if not (0.0 <= adaptation.learning_rate_global <= 1.0):
            self.errors.append("runtime_adaptation.learning_rate_global must be between 0 and 1")
        
        # Validate adaptive parameters
        for param_name, param_config in adaptation.adaptive_parameters.items():
            if not isinstance(param_config, dict):
                self.errors.append(f"Adaptive parameter '{param_name}' must be a dictionary")
                continue
            
            if "bounds" in param_config:
                bounds = param_config["bounds"]
                if not isinstance(bounds, list) or len(bounds) != 2:
                    self.errors.append(f"Parameter '{param_name}' bounds must be [min, max]")
                elif bounds[0] >= bounds[1]:
                    self.errors.append(f"Parameter '{param_name}' min bound must be less than max bound")
    
    def _validate_memory_profile(self, memory_profile: MemoryProfile):
        """Validate memory profile configuration"""
        if memory_profile.max_memory_gb <= 0:
            self.errors.append("memory_profile.max_memory_gb must be positive")
        
        # Validate tier allocations sum to ~1.0
        tier_sum = sum(memory_profile.tier_allocation.values())
        if abs(tier_sum - 1.0) > 0.01:
            self.warnings.append(f"Memory tier allocations sum to {tier_sum:.3f}, should be ~1.0")
    
    def _validate_platform_translation(self, translation: PlatformTranslation):
        """Validate platform translation configuration"""
        # Basic validation - could be expanded
        if not isinstance(translation.ollama, dict):
            self.errors.append("platform_translation.ollama must be a dictionary")
        
        if not isinstance(translation.gguf, dict):
            self.errors.append("platform_translation.gguf must be a dictionary")


class CASParser:
    """Parses CAS YAML files into structured objects"""
    
    def __init__(self):
        self.validator = CASValidator()
    
    def parse_file(self, cas_file_path: Union[str, Path]) -> Tuple[Optional[CASSpecification], List[str], List[str]]:
        """Parse CAS file from disk"""
        cas_path = Path(cas_file_path)
        
        if not cas_path.exists():
            return None, [f"CAS file not found: {cas_path}"], []
        
        try:
            with open(cas_path, 'r', encoding='utf-8') as f:
                cas_data = yaml.safe_load(f)
            
            return self.parse_dict(cas_data)
            
        except Exception as e:
            return None, [f"Failed to parse CAS file: {e}"], []
    
    def parse_dict(self, cas_data: Dict[str, Any]) -> Tuple[Optional[CASSpecification], List[str], List[str]]:
        """Parse CAS data from dictionary"""
        try:
            # Parse metadata
            metadata_data = cas_data.get('metadata', {})
            metadata = CASMetadata(
                system_model_id=metadata_data.get('system_model_id', ''),
                custom_name=metadata_data.get('custom_name', ''),
                version=metadata_data.get('version', '1.0.0'),
                author=metadata_data.get('author', 'Unknown'),
                description=metadata_data.get('description', ''),
                created_at=metadata_data.get('created_at', ''),
                cas_version=metadata_data.get('cas_version', '1.0'),
                sha256_hash=metadata_data.get('sha256_hash', ''),
                project_context=metadata_data.get('project_context', '')
            )
            
            # Parse base model
            base_model_data = cas_data.get('base_model', {})
            base_model = BaseModelConfig(
                provider=base_model_data.get('provider', 'local'),
                path=base_model_data.get('path', ''),
                type=base_model_data.get('type', 'gguf'),
                architecture=base_model_data.get('architecture', 'llama'),
                parameter_count=base_model_data.get('parameter_count', '7B'),
                context_length=base_model_data.get('context_length', 8192),
                quantization=base_model_data.get('quantization')
            )
            
            # Parse constitutional framework
            const_data = cas_data.get('constitutional_framework', {})
            constitutional_framework = ConstitutionalFramework(
                governor_mode=const_data.get('governor_mode', 'balanced'),
                enforcement_level=const_data.get('enforcement_level', 'warn_and_proceed'),
                safety_principles=const_data.get('safety_principles', {}),
                content_filters=const_data.get('content_filters', {}),
                override_permissions=const_data.get('override_permissions', {})
            )
            
            # Parse cognitive architecture (keep as dict for flexibility)
            cognitive_architecture = cas_data.get('cognitive_architecture', {})
            
            # Parse runtime adaptation
            adapt_data = cas_data.get('runtime_adaptation', {})
            runtime_adaptation = RuntimeAdaptation(
                enabled=adapt_data.get('enabled', True),
                adaptation_frequency=adapt_data.get('adaptation_frequency', 25),
                learning_rate_global=adapt_data.get('learning_rate_global', 0.02),
                adaptive_parameters=adapt_data.get('adaptive_parameters', {}),
                adaptation_rules=adapt_data.get('adaptation_rules', []),
                adaptation_constraints=adapt_data.get('adaptation_constraints', {})
            )
            
            # Parse memory profile
            memory_data = cas_data.get('memory_profile', {})
            memory_profile = MemoryProfile(
                max_memory_gb=memory_data.get('max_memory_gb', 6.0),
                memory_management=memory_data.get('memory_management', 'adaptive'),
                tier_allocation=memory_data.get('tier_allocation', {}),
                caching_policies=memory_data.get('caching_policies', []),
                compression_config=memory_data.get('compression_config', {}),
                optimization_triggers=memory_data.get('optimization_triggers', [])
            )
            
            # Parse platform translation
            platform_data = cas_data.get('platform_translation', {})
            platform_translation = PlatformTranslation(
                ollama=platform_data.get('ollama', {}),
                gguf=platform_data.get('gguf', {})
            )
            
            # Create complete specification
            cas_spec = CASSpecification(
                metadata=metadata,
                base_model=base_model,
                constitutional_framework=constitutional_framework,
                cognitive_architecture=cognitive_architecture,
                runtime_adaptation=runtime_adaptation,
                memory_profile=memory_profile,
                platform_translation=platform_translation,
                development_config=cas_data.get('development_config', {}),
                extensions=cas_data.get('extensions', {})
            )
            
            # Validate the specification
            is_valid, errors, warnings = self.validator.validate(cas_spec)
            
            if is_valid:
                return cas_spec, [], warnings
            else:
                return cas_spec, errors, warnings
                
        except Exception as e:
            return None, [f"Failed to parse CAS data: {e}"], []


class CASGenerator:
    """Generates CAS files from templates and configurations"""
    
    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        self.template_dir = Path(template_dir) if template_dir else Path("./cas_templates")
        self.template_dir.mkdir(exist_ok=True)
        self.config_dir = Path("./config")
        self.template_cache: Dict[str, str] = {}
        self.template_validator = TemplateValidator()
        self.debug_mode = False
        self.generation_history: List[Dict[str, Any]] = []
        self._validate_environment()
    
    def _validate_environment(self) -> None:
        """Validate that required template files exist"""
        cas_spec_path = self.config_dir / "cas_specification.yaml"
        if not cas_spec_path.exists():
            raise FileNotFoundError(f"CAS specification template not found: {cas_spec_path}")
    
    def generate_from_template(self, 
                             template_name: str,
                             model_name: str,
                             model_path: str,
                             **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Generate CAS YAML from template with comprehensive validation and debugging"""
        
        generation_start_time = time.time()
        generation_id = hashlib.md5(f"{template_name}_{model_name}_{time.time()}".encode()).hexdigest()[:8]
        
        debug_info = {
            'generation_id': generation_id,
            'template_name': template_name,
            'model_name': model_name,
            'model_path': model_path,
            'kwargs': kwargs.copy(),
            'start_time': generation_start_time,
            'validation_steps': []
        }
        
        try:
            # Enhanced input validation
            validation_errors = self._validate_generation_inputs(template_name, model_name, model_path, **kwargs)
            if validation_errors:
                debug_info['input_validation_errors'] = validation_errors
                raise ValueError(f"Input validation failed: {'; '.join(validation_errors)}")
            
            debug_info['validation_steps'].append({'step': 'input_validation', 'status': 'passed'})
            
            # Load template content with validation
            try:
                template_content = self._load_template(template_name)
                debug_info['template_loaded'] = True
                debug_info['template_size'] = len(template_content)
            except Exception as e:
                debug_info['template_load_error'] = str(e)
                logger.error(f"Failed to load template '{template_name}': {e}")
                raise ValueError(f"Invalid template: {template_name}") from e
            
            # Validate template content structure and syntax
            is_valid, template_errors, template_warnings = self.template_validator.validate_template_content(template_content)
            debug_info['template_validation'] = {
                'valid': is_valid,
                'errors': template_errors,
                'warnings': template_warnings
            }
            
            if not is_valid:
                logger.error(f"Template validation failed: {template_errors}")
                raise ValueError(f"Invalid template structure: {'; '.join(template_errors)}")
            
            if template_warnings and self.debug_mode:
                logger.warning(f"Template warnings: {template_warnings}")
            
            debug_info['validation_steps'].append({'step': 'template_validation', 'status': 'passed'})
            
            # Prepare template variables with enhanced validation
            template_vars = self._prepare_template_variables(model_name, model_path, **kwargs)
            debug_info['template_vars'] = template_vars.copy()
            
            # Validate template variables against schemas
            vars_valid, var_errors, var_warnings = self.template_validator.validate_template_variables(template_vars)
            debug_info['variable_validation'] = {
                'valid': vars_valid,
                'errors': var_errors,
                'warnings': var_warnings
            }
            
            if not vars_valid:
                logger.error(f"Variable validation failed: {var_errors}")
                raise ValueError(f"Invalid template variables: {'; '.join(var_errors)}")
            
            if var_warnings and self.debug_mode:
                logger.warning(f"Variable warnings: {var_warnings}")
            
            debug_info['validation_steps'].append({'step': 'variable_validation', 'status': 'passed'})
            
            # Perform template substitution with enhanced error reporting
            try:
                filled_content, substitution_info = self._substitute_template_variables_enhanced(template_content, template_vars)
                debug_info['substitution_info'] = substitution_info
            except Exception as e:
                debug_info['substitution_error'] = str(e)
                logger.error(f"Template substitution failed: {e}")
                raise ValueError("Template variable substitution failed") from e
            
            debug_info['validation_steps'].append({'step': 'template_substitution', 'status': 'passed'})
            
            # Calculate and insert content hash
            cas_hash = hashlib.sha256(filled_content.encode('utf-8')).hexdigest()
            template_vars['cas_file_hash'] = cas_hash
            filled_content = filled_content.replace("{cas_file_hash}", cas_hash)
            debug_info['content_hash'] = cas_hash
            
            # Final validation of generated content
            try:
                self._validate_generated_content_enhanced(filled_content)
                debug_info['final_validation'] = {'status': 'passed'}
            except Exception as e:
                debug_info['final_validation'] = {'status': 'failed', 'error': str(e)}
                logger.error(f"Final content validation failed: {e}")
                raise
            
            debug_info['validation_steps'].append({'step': 'final_validation', 'status': 'passed'})
            
            # Calculate generation metrics
            generation_time = time.time() - generation_start_time
            debug_info['generation_time_ms'] = round(generation_time * 1000, 2)
            debug_info['content_size'] = len(filled_content)
            debug_info['variables_count'] = len(template_vars)
            
            # Store generation history for debugging
            self.generation_history.append(debug_info)
            if len(self.generation_history) > 100:  # Keep last 100 generations
                self.generation_history.pop(0)
            
            logger.info(f"Generated CAS content for model: {model_name} (ID: {generation_id}, Time: {debug_info['generation_time_ms']}ms)")
            
            return filled_content, {**template_vars, '_debug_info': debug_info}
            
        except Exception as e:
            debug_info['error'] = str(e)
            debug_info['generation_time_ms'] = round((time.time() - generation_start_time) * 1000, 2)
            self.generation_history.append(debug_info)
            raise
    
    def _load_template(self, template_name: str) -> str:
        """Load template content from file with caching"""
        if template_name in self.template_cache:
            return self.template_cache[template_name]
        
        # Load the real CAS specification template
        cas_spec_path = self.config_dir / "cas_specification.yaml"
        
        try:
            with open(cas_spec_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Cache the template for performance
            self.template_cache[template_name] = template_content
            return template_content
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {cas_spec_path}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading template: {cas_spec_path}")
        except UnicodeDecodeError:
            raise ValueError(f"Invalid encoding in template file: {cas_spec_path}")
    
    def _prepare_template_variables(self, model_name: str, model_path: str, **kwargs) -> Dict[str, Any]:
        """Prepare template variables with validation and defaults"""
        
        # Generate system model ID
        system_model_id = self._generate_system_model_id()
        
        # Current timestamp
        created_at = datetime.now(timezone.utc).isoformat()
        
        # Prepare base variables
        template_vars = {
            'system_model_id': system_model_id,
            'model_name': self._sanitize_string(model_name),
            'model_path': self._sanitize_path(model_path),
            'created_by': self._sanitize_string(kwargs.get('created_by', 'CAS Generator')),
            'description': self._sanitize_string(kwargs.get('description', f'Generated CAS for {model_name}')),
            'created_at': created_at,
            'project_name': self._sanitize_string(kwargs.get('project_name', 'default')),
            'model_type': self._validate_model_type(kwargs.get('model_type', 'gguf')),
            'parameter_count': self._validate_parameter_count(kwargs.get('parameter_count', '7B')),
            'cas_file_hash': 'PLACEHOLDER',  # Will be replaced after content generation
        }
        
        # Add validated optional parameters
        optional_params = {
            'cognitive_profile': kwargs.get('cognitive_profile', 'analytical'),
            'safety_mode': kwargs.get('safety_mode', 'balanced'),
            'context_length': kwargs.get('context_length', 8192),
            'quantization': kwargs.get('quantization', 'Q4_K_M'),
            'architecture': kwargs.get('architecture', 'llama'),
            'provider': kwargs.get('provider', 'local'),
        }
        
        # Validate optional parameters
        for key, value in optional_params.items():
            template_vars[key] = self._validate_optional_param(key, value)
        
        return template_vars
    
    def _substitute_template_variables(self, template_content: str, variables: Dict[str, Any]) -> str:
        """Perform template variable substitution with validation"""
        
        filled_content = template_content
        substituted_vars: Set[str] = set()
        
        # Find all template variables in content
        template_pattern = re.compile(r'\{([^}]+)\}')
        found_vars = set(template_pattern.findall(filled_content))
        
        # Substitute each variable
        for var_name, value in variables.items():
            placeholder = f"{{{var_name}}}"
            if placeholder in filled_content:
                filled_content = filled_content.replace(placeholder, str(value))
                substituted_vars.add(var_name)
        
        # Check for unsubstituted variables (excluding hash placeholder)
        remaining_vars = found_vars - substituted_vars - {'cas_file_hash'}
        if remaining_vars:
            logger.warning(f"Unsubstituted template variables: {remaining_vars}")
        
        return filled_content
    
    def _validate_generated_content(self, content: str) -> None:
        """Validate that generated content is valid YAML"""
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Generated content is not valid YAML: {e}")
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input for YAML safety"""
        if not isinstance(value, str):
            value = str(value)
        
        # Remove control characters and limit length
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        return sanitized[:256]  # Reasonable length limit
    
    def _sanitize_path(self, path: str) -> str:
        """Sanitize file path for security"""
        if not isinstance(path, str):
            path = str(path)
        
        # Normalize path and check for path traversal attempts
        normalized = Path(path).as_posix()
        if '..' in normalized or normalized.startswith('/'):
            logger.warning(f"Suspicious path detected: {path}")
        
        return normalized
    
    def _validate_model_type(self, model_type: str) -> str:
        """Validate model type against allowed values"""
        valid_types = [t.value for t in ModelType]
        if model_type not in valid_types:
            logger.warning(f"Unknown model type '{model_type}', using 'gguf'")
            return 'gguf'
        return model_type
    
    def _validate_parameter_count(self, param_count: str) -> str:
        """Validate parameter count format"""
        if not isinstance(param_count, str):
            param_count = str(param_count)
        
        # Check for common parameter count patterns
        valid_pattern = re.compile(r'^\d+\.?\d*[BMK]?$', re.IGNORECASE)
        if not valid_pattern.match(param_count):
            logger.warning(f"Unusual parameter count format: {param_count}")
        
        return param_count
    
    def _validate_optional_param(self, param_name: str, value: Any) -> Any:
        """Validate optional parameters based on type"""
        if param_name == 'context_length':
            if not isinstance(value, int) or value <= 0:
                logger.warning(f"Invalid context_length: {value}, using 8192")
                return 8192
        elif param_name in ['cognitive_profile', 'safety_mode', 'architecture', 'provider']:
            return self._sanitize_string(str(value))
        elif param_name == 'quantization' and value is not None:
            return self._sanitize_string(str(value))
        
        return value
    
    def _generate_system_model_id(self) -> str:
        """Generate unique system model ID using timestamp and entropy"""
        timestamp = int(time.time() * 1000)  # Millisecond precision
        entropy_bytes = hashlib.sha256(str(time.time()).encode()).digest()[:4]
        entropy_hex = entropy_bytes.hex()
        return f"cas_model_{timestamp}_{entropy_hex}"
    
    def _validate_generation_inputs(self, template_name: str, model_name: str, model_path: str, **kwargs) -> List[str]:
        """Enhanced input validation for template generation"""
        errors = []
        
        # Template name validation
        if not template_name or not isinstance(template_name, str):
            errors.append("template_name must be a non-empty string")
        elif len(template_name) > 64:
            errors.append("template_name is too long (max 64 characters)")
        elif not re.match(r'^[a-zA-Z0-9_-]+$', template_name):
            errors.append("template_name contains invalid characters")
        
        # Model name validation
        if not model_name or not isinstance(model_name, str):
            errors.append("model_name must be a non-empty string")
        elif len(model_name) > 128:
            errors.append("model_name is too long (max 128 characters)")
        elif not re.match(r'^[a-zA-Z0-9\s_-]+$', model_name):
            errors.append("model_name contains invalid characters")
        
        # Model path validation
        if not model_path or not isinstance(model_path, str):
            errors.append("model_path must be a non-empty string")
        elif len(model_path) > 512:
            errors.append("model_path is too long (max 512 characters)")
        elif '..' in model_path:
            errors.append("model_path contains path traversal sequences")
        
        # Validate optional parameters
        for key, value in kwargs.items():
            if key == 'context_length':
                if not isinstance(value, int) or value < 512 or value > 131072:
                    errors.append(f"context_length must be an integer between 512 and 131072")
            elif key == 'model_type':
                valid_types = ['gguf', 'pytorch', 'onnx', 'bitnet', 'safetensors']
                if value not in valid_types:
                    errors.append(f"model_type must be one of: {', '.join(valid_types)}")
            elif key == 'cognitive_profile':
                valid_profiles = ['analytical', 'creative', 'conversational', 'research', 'technical', 'teaching']
                if value not in valid_profiles:
                    errors.append(f"cognitive_profile must be one of: {', '.join(valid_profiles)}")
            elif key == 'safety_mode':
                valid_modes = ['strict', 'balanced', 'permissive', 'disabled']
                if value not in valid_modes:
                    errors.append(f"safety_mode must be one of: {', '.join(valid_modes)}")
        
        return errors
    
    def _substitute_template_variables_enhanced(self, template_content: str, variables: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Enhanced template variable substitution with detailed reporting"""
        
        filled_content = template_content
        substituted_vars: Set[str] = set()
        failed_substitutions: List[str] = []
        substitution_info = {
            'total_variables': len(variables),
            'substituted_count': 0,
            'failed_count': 0,
            'unsubstituted_variables': []
        }
        
        # Find all template variables in content
        template_pattern = re.compile(r'\{([^}]+)\}')
        found_vars = set(template_pattern.findall(filled_content))
        substitution_info['found_in_template'] = list(found_vars)
        
        # Substitute each variable with error handling
        for var_name, value in variables.items():
            placeholder = f"{{{var_name}}}"
            if placeholder in filled_content:
                try:
                    # Convert value to string safely
                    str_value = self._safe_value_conversion(value)
                    filled_content = filled_content.replace(placeholder, str_value)
                    substituted_vars.add(var_name)
                except Exception as e:
                    failed_substitutions.append(f"{var_name}: {str(e)}")
        
        # Check for unsubstituted variables (excluding hash placeholder)
        remaining_vars = found_vars - substituted_vars - {'cas_file_hash'}
        
        substitution_info.update({
            'substituted_count': len(substituted_vars),
            'failed_count': len(failed_substitutions),
            'unsubstituted_variables': list(remaining_vars),
            'failed_substitutions': failed_substitutions
        })
        
        if failed_substitutions:
            raise ValueError(f"Variable substitution failures: {'; '.join(failed_substitutions)}")
        
        if remaining_vars and self.debug_mode:
            logger.warning(f"Unsubstituted template variables: {remaining_vars}")
        
        return filled_content, substitution_info
    
    def _safe_value_conversion(self, value: Any) -> str:
        """Safely convert value to string for template substitution"""
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (list, dict)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Cannot serialize complex value: {e}")
        else:
            return str(value)
    
    def _validate_generated_content_enhanced(self, content: str) -> None:
        """Enhanced validation of generated content"""
        
        # Basic YAML validation
        try:
            parsed_content = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Generated content is not valid YAML: {e}")
        
        # Structure validation
        required_sections = ['metadata', 'base_model', 'constitutional_framework', 'cognitive_architecture']
        for section in required_sections:
            if section not in parsed_content:
                raise ValueError(f"Generated content missing required section: {section}")
        
        # Content size validation
        if len(content) > 1000000:  # 1MB limit
            raise ValueError("Generated content is too large (>1MB)")
        
        # Security validation - check for potential injection
        security_patterns = [
            r'<%.*?%>',  # Template execution
            r'\{\{.*?\}\}',  # Jinja2 execution
            r'exec\s*\(',  # Python exec
            r'__import__',  # Dynamic imports
        ]
        
        for pattern in security_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                raise ValueError(f"Generated content contains potential security risk: {pattern}")
    
    def enable_debug_mode(self) -> None:
        """Enable debug mode for detailed logging and validation"""
        self.debug_mode = True
        logger.info("CAS Generator debug mode enabled")
    
    def disable_debug_mode(self) -> None:
        """Disable debug mode"""
        self.debug_mode = False
        logger.info("CAS Generator debug mode disabled")
    
    def get_generation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent generation history for debugging"""
        return self.generation_history[-limit:] if limit else self.generation_history.copy()
    
    def get_template_documentation(self) -> Dict[str, Any]:
        """Get comprehensive template documentation"""
        return {
            'supported_variables': self.template_validator.get_variable_documentation(),
            'template_structure': {
                'required_sections': ['metadata', 'base_model', 'constitutional_framework', 'cognitive_architecture'],
                'optional_sections': ['runtime_adaptation', 'memory_profile', 'platform_translation', 'development_config', 'extensions']
            },
            'validation_rules': {
                'variable_naming': 'Must match ^[a-zA-Z_][a-zA-Z0-9_]*$ pattern',
                'template_size_limit': '100KB recommended',
                'max_nesting_depth': '8 levels',
                'security_restrictions': 'No code execution patterns allowed'
            }
        }
    
    def validate_template_file(self, template_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate a template file and return detailed results"""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            is_valid, errors, warnings = self.template_validator.validate_template_content(content)
            
            return {
                'file_path': str(template_path),
                'valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'size_bytes': len(content),
                'variable_count': len(self.template_validator._extract_template_variables(content)),
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'file_path': str(template_path),
                'valid': False,
                'errors': [f"Failed to read template file: {e}"],
                'warnings': [],
                'timestamp': time.time()
            }

    def save_cas_file(self, 
                     cas_content: str, 
                     output_path: Union[str, Path]) -> Path:
        """Save CAS content to file with comprehensive error handling and validation"""
        
        # Input validation
        if not cas_content or not isinstance(cas_content, str):
            raise ValueError("cas_content must be a non-empty string")
        
        if not output_path:
            raise ValueError("output_path must be provided")
        
        # Convert to Path object and validate
        try:
            output_path = Path(output_path)
        except Exception as e:
            raise ValueError(f"Invalid output path: {e}")
        
        # Security validation - prevent path traversal
        if '..' in output_path.parts:
            raise ValueError("Path traversal detected in output_path")
        
        # Validate file extension
        if not output_path.suffix.lower() in ['.yml', '.yaml', '.cas']:
            logger.warning(f"Unusual file extension for CAS file: {output_path.suffix}")
        
        # Validate content is valid YAML before saving
        try:
            yaml.safe_load(cas_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Content is not valid YAML: {e}")
        
        # Create parent directories with proper error handling
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Permission denied creating directory: {output_path.parent}")
        except OSError as e:
            raise OSError(f"Failed to create directory {output_path.parent}: {e}")
        
        # Check if file already exists and handle appropriately
        if output_path.exists():
            logger.info(f"Overwriting existing CAS file: {output_path}")
            
            # Create backup of existing file
            backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
            try:
                shutil.copy2(output_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        # Write content with atomic operation (write to temp file, then rename)
        temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        
        try:
            # Write to temporary file
            with open(temp_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(cas_content)
                f.flush()  # Ensure data is written to disk
                
            # Verify written content
            with open(temp_path, 'r', encoding='utf-8') as f:
                written_content = f.read()
                
            if written_content != cas_content:
                raise IOError("Content verification failed after write")
            
            # Atomic rename to final location
            temp_path.replace(output_path)
            
        except PermissionError:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise PermissionError(f"Permission denied writing to: {output_path}")
            
        except OSError as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise OSError(f"Failed to write CAS file: {e}")
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Unexpected error saving CAS file: {e}")
        
        # Verify final file
        try:
            if not output_path.exists():
                raise FileNotFoundError("File was not created successfully")
            
            file_size = output_path.stat().st_size
            if file_size == 0:
                raise ValueError("Generated file is empty")
            
            # Verify content can still be parsed
            with open(output_path, 'r', encoding='utf-8') as f:
                verification_content = f.read()
            yaml.safe_load(verification_content)
            
        except Exception as e:
            logger.error(f"File verification failed: {e}")
            # Try to clean up the invalid file
            try:
                output_path.unlink()
                logger.warning(f"Removed failed CAS file: {output_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup failed CAS file {output_path}: {cleanup_error}")
            raise RuntimeError(f"File verification failed: {e}")
        
        # Calculate and log file hash for integrity tracking
        try:
            file_hash = hashlib.sha256(cas_content.encode('utf-8')).hexdigest()
            logger.info(f"Generated CAS file: {output_path}")
            logger.info(f"File size: {file_size} bytes")
            logger.info(f"Content hash: {file_hash[:16]}...")
        except Exception as e:
            logger.warning(f"Failed to calculate file hash: {e}")
        
        return output_path


class ConstitutionalGovernor:
    """Advanced constitutional AI safety enforcement with sophisticated analysis"""
    
    def __init__(self, constitutional_framework: ConstitutionalFramework):
        self.framework = constitutional_framework
        self.violation_count = 0
        self.violation_log: deque = deque(maxlen=1000)  # Circular buffer for performance
        self.pattern_cache: Dict[str, re.Pattern] = {}
        self.context_history: deque = deque(maxlen=50)  # Context awareness
        self.behavioral_metrics = defaultdict(int)
        self.risk_scores = defaultdict(list)
        self._lock = threading.RLock()
        
        # Initialize advanced safety patterns
        self._initialize_safety_patterns()
        
        # Performance metrics
        self.analysis_count = 0
        self.total_analysis_time = 0.0
        
    def _initialize_safety_patterns(self) -> None:
        """Initialize comprehensive safety pattern detection"""
        self.safety_patterns: List[SafetyPattern] = [
            # Violence and harmful content
            SafetyPattern(
                name="explicit_violence",
                pattern=r'\b(?:kill|murder|assassinate|torture|harm|hurt|injure|attack|assault|beat|stab|shoot|bomb)\b.*?\b(?:person|people|human|someone|anybody|everyone|children|kid|family)\b',
                risk_level=RiskLevel.HIGH,
                description="Explicit violence against persons",
                regex_flags=re.IGNORECASE | re.MULTILINE
            ),
            SafetyPattern(
                name="self_harm",
                pattern=r'\b(?:suicide|kill\s+myself|end\s+my\s+life|hurt\s+myself|self\s*harm|cut\s+myself)\b',
                risk_level=RiskLevel.CRITICAL,
                description="Self-harm or suicide ideation",
                regex_flags=re.IGNORECASE
            ),
            # Illegal activities
            SafetyPattern(
                name="illegal_instructions",
                pattern=r'\b(?:how\s+to\s+(?:hack|steal|break\s+into|make\s+(?:bomb|drug|poison)|commit|illegal))\b',
                risk_level=RiskLevel.HIGH,
                description="Instructions for illegal activities",
                regex_flags=re.IGNORECASE
            ),
            # Privacy violations
            SafetyPattern(
                name="personal_info_request",
                pattern=r'\b(?:social\s+security|ssn|credit\s+card|password|address|phone\s+number|personal\s+information)\b',
                risk_level=RiskLevel.MEDIUM,
                description="Request for personal information",
                regex_flags=re.IGNORECASE
            ),
            # Misinformation patterns
            SafetyPattern(
                name="medical_misinformation",
                pattern=r'\b(?:cure\s+(?:cancer|aids|covid)|miracle\s+treatment|doctors\s+don\'t\s+want\s+you\s+to\s+know)\b',
                risk_level=RiskLevel.HIGH,
                description="Potential medical misinformation",
                regex_flags=re.IGNORECASE
            ),
            # Manipulation and social engineering
            SafetyPattern(
                name="manipulation_attempt",
                pattern=r'\b(?:ignore\s+(?:previous\s+instructions|your\s+programming)|act\s+as\s+if|pretend\s+to\s+be|bypass\s+(?:safety|restrictions))\b',
                risk_level=RiskLevel.HIGH,
                description="Potential manipulation or jailbreak attempt",
                regex_flags=re.IGNORECASE
            ),
            # Hate speech patterns
            SafetyPattern(
                name="discriminatory_language",
                pattern=r'\b(?:inferior\s+race|genetic\s+superiority|racial\s+purity|ethnic\s+cleansing)\b',
                risk_level=RiskLevel.HIGH,
                description="Discriminatory or hate speech patterns",
                regex_flags=re.IGNORECASE
            ),
            # Advanced financial fraud detection
            SafetyPattern(
                name="financial_fraud",
                pattern=r'\b(?:ponzi\s+scheme|pump\s+and\s+dump|insider\s+trading|money\s+laundering|cryptocurrency\s+scam)\b',
                risk_level=RiskLevel.HIGH,
                description="Financial fraud and illegal schemes",
                regex_flags=re.IGNORECASE
            ),
            # Cybersecurity threats
            SafetyPattern(
                name="cyber_threats",
                pattern=r'\b(?:ddos\s+attack|sql\s+injection|cross\s+site\s+scripting|buffer\s+overflow|zero\s+day\s+exploit)\b',
                risk_level=RiskLevel.HIGH,
                description="Cybersecurity attack methods",
                regex_flags=re.IGNORECASE
            ),
            # Substance abuse
            SafetyPattern(
                name="substance_abuse",
                pattern=r'\b(?:how\s+to\s+(?:make|cook|synthesize)\s+(?:meth|cocaine|heroin|fentanyl|lsd))\b',
                risk_level=RiskLevel.CRITICAL,
                description="Drug manufacturing instructions",
                regex_flags=re.IGNORECASE
            ),
            # Social engineering
            SafetyPattern(
                name="social_engineering",
                pattern=r'\b(?:phishing\s+email|pretexting|baiting|quid\s+pro\s+quo|tailgating\s+attack)\b',
                risk_level=RiskLevel.MEDIUM,
                description="Social engineering tactics",
                regex_flags=re.IGNORECASE
            ),
            # Advanced manipulation detection
            SafetyPattern(
                name="advanced_manipulation",
                pattern=r'\b(?:gaslight|emotional\s+manipulation|love\s+bombing|triangulation|silent\s+treatment)\b',
                risk_level=RiskLevel.MEDIUM,
                description="Psychological manipulation tactics",
                regex_flags=re.IGNORECASE
            ),
            # Conspiracy theories and misinformation
            SafetyPattern(
                name="conspiracy_misinformation",
                pattern=r'\b(?:qanon|flat\s+earth|vaccine\s+microchip|5g\s+coronavirus|chemtrails|lizard\s+people)\b',
                risk_level=RiskLevel.MEDIUM,
                description="Common conspiracy theories and misinformation",
                regex_flags=re.IGNORECASE
            ),
            # Child safety
            SafetyPattern(
                name="child_safety",
                pattern=r'\b(?:child\s+(?:abuse|exploitation|grooming)|underage\s+(?:content|material))\b',
                risk_level=RiskLevel.CRITICAL,
                description="Child safety violations",
                regex_flags=re.IGNORECASE
            )
        ]
        
        # Compile patterns for performance
        for pattern in self.safety_patterns:
            if pattern.enabled:
                try:
                    self.pattern_cache[pattern.name] = re.compile(
                        pattern.pattern, 
                        pattern.regex_flags
                    )
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern.name}': {e}")
    
    def check_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str], Optional[ViolationAnalysis]]:
        """Enhanced input checking with context awareness"""
        start_time = time.time()
        
        try:
            with self._lock:
                self.analysis_count += 1
                
                if self.framework.governor_mode == "disabled":
                    return True, None, None
                
                # Store context for pattern analysis
                if context:
                    self.context_history.append({
                        'timestamp': time.time(),
                        'type': 'input_context',
                        'data': context
                    })
                
                # Comprehensive analysis
                analysis = self._analyze_content(
                    content=user_input,
                    content_type='input',
                    context=context
                )
                
                # Update behavioral metrics
                self._update_behavioral_metrics(user_input, analysis)
                
                # Process violations
                if analysis:
                    return self._process_violations(analysis, user_input[:200])
                
                return True, None, None
                
        except Exception as e:
            logger.error(f"Error in input checking: {e}")
            # Fail safe - allow content but log error
            return True, f"Safety check error: {str(e)}", None
        finally:
            self.total_analysis_time += time.time() - start_time
    
    def check_output(self, model_output: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str], Optional[ViolationAnalysis]]:
        """Enhanced output checking with context awareness"""
        start_time = time.time()
        
        try:
            with self._lock:
                self.analysis_count += 1
                
                if self.framework.governor_mode == "disabled":
                    return True, None, None
                
                # Comprehensive analysis
                analysis = self._analyze_content(
                    content=model_output,
                    content_type='output',
                    context=context
                )
                
                # Check for behavioral anomalies in output
                anomaly_analysis = self._detect_behavioral_anomalies(model_output)
                if anomaly_analysis:
                    if analysis:
                        # Merge analyses
                        analysis.matched_patterns.extend(anomaly_analysis.matched_patterns)
                        analysis.risk_level = max(analysis.risk_level, anomaly_analysis.risk_level, key=lambda x: x.value)
                    else:
                        analysis = anomaly_analysis
                
                # Process violations
                if analysis:
                    return self._process_violations(analysis, model_output[:200])
                
                return True, None, None
                
        except Exception as e:
            logger.error(f"Error in output checking: {e}")
            # Fail safe - block output on error for safety
            return False, f"Safety check error - output blocked: {str(e)}", None
        finally:
            self.total_analysis_time += time.time() - start_time
    
    def _analyze_content(self, content: str, content_type: str, context: Optional[Dict[str, Any]] = None) -> Optional[ViolationAnalysis]:
        """Comprehensive content analysis using multiple detection methods"""
        
        # Pattern-based detection
        pattern_violations = self._check_safety_patterns(content)
        
        # Content filter violations
        filter_violations = self._check_enhanced_content_filters(content)
        
        # Safety principle violations
        principle_violations = self._check_enhanced_safety_principles(content, context)
        
        # Context-aware violations
        context_violations = self._check_context_violations(content, context)
        
        # Combine all violations
        all_violations = pattern_violations + filter_violations + principle_violations + context_violations
        
        if not all_violations:
            return None
        
        # Determine overall risk level and confidence
        risk_levels = [v.risk_level for v in all_violations]
        highest_risk = max(risk_levels, key=lambda x: ['low', 'medium', 'high', 'critical'].index(x.value))
        
        # Calculate confidence based on number and strength of matches
        confidence = min(1.0, len(all_violations) * 0.2 + max(v.confidence for v in all_violations))
        
        # Extract context window around violations
        context_window = self._extract_context_window(content, all_violations)
        
        # Determine suggested action
        suggested_action = self._determine_suggested_action(highest_risk, confidence)
        
        return ViolationAnalysis(
            violation_type=ViolationType.PATTERN_DETECTION,
            risk_level=highest_risk,
            confidence=confidence,
            matched_patterns=[v.matched_patterns[0] if v.matched_patterns else 'unknown' for v in all_violations],
            context_window=context_window,
            suggested_action=suggested_action
        )
    
    def _check_safety_patterns(self, content: str) -> List[ViolationAnalysis]:
        """Check content against compiled safety patterns"""
        violations = []
        
        for pattern in self.safety_patterns:
            if not pattern.enabled or pattern.name not in self.pattern_cache:
                continue
            
            regex = self.pattern_cache[pattern.name]
            matches = regex.finditer(content)
            
            for match in matches:
                confidence = self._calculate_pattern_confidence(match, content)
                
                violations.append(ViolationAnalysis(
                    violation_type=ViolationType.PATTERN_DETECTION,
                    risk_level=pattern.risk_level,
                    confidence=confidence,
                    matched_patterns=[pattern.name],
                    context_window=match.group(0),
                    suggested_action=self._determine_suggested_action(pattern.risk_level, confidence)
                ))
        
        return violations
    
    def _check_enhanced_content_filters(self, content: str) -> List[ViolationAnalysis]:
        """Enhanced content filtering with severity assessment"""
        violations = []
        content_lower = content.lower()
        
        # Violence detection with severity levels
        violence_config = self.framework.content_filters.get('violence', 'disabled')
        if violence_config != 'disabled':
            violence_patterns = {
                'extreme': ['massacre', 'genocide', 'torture', 'execution'],
                'high': ['kill', 'murder', 'assassinate', 'attack'],
                'medium': ['fight', 'beat', 'punch', 'slap'],
                'low': ['conflict', 'aggressive', 'hostile']
            }
            
            for severity, keywords in violence_patterns.items():
                if any(keyword in content_lower for keyword in keywords):
                    risk_map = {'extreme': RiskLevel.CRITICAL, 'high': RiskLevel.HIGH, 
                               'medium': RiskLevel.MEDIUM, 'low': RiskLevel.LOW}
                    
                    violations.append(ViolationAnalysis(
                        violation_type=ViolationType.CONTENT_FILTER,
                        risk_level=risk_map[severity],
                        confidence=0.8,
                        matched_patterns=[f'violence_{severity}'],
                        context_window=content[:100],
                        suggested_action=self._determine_suggested_action(risk_map[severity], 0.8)
                    ))
                    break  # Only report highest severity
        
        # Enhanced illegal activity detection
        illegal_config = self.framework.content_filters.get('illegal_activities', 'disabled')
        if illegal_config != 'disabled':
            illegal_categories = {
                'drug_manufacturing': ['make meth', 'cook drugs', 'synthesize'],
                'hacking': ['hack into', 'break encryption', 'ddos'],
                'fraud': ['credit card fraud', 'identity theft', 'scam'],
                'weapons': ['make bomb', 'build weapon', 'explosive device']
            }
            
            for category, patterns in illegal_categories.items():
                if any(pattern in content_lower for pattern in patterns):
                    violations.append(ViolationAnalysis(
                        violation_type=ViolationType.CONTENT_FILTER,
                        risk_level=RiskLevel.HIGH,
                        confidence=0.9,
                        matched_patterns=[f'illegal_{category}'],
                        context_window=content[:100],
                        suggested_action="BLOCK"
                    ))
        
        return violations
    
    def _check_enhanced_safety_principles(self, content: str, context: Optional[Dict[str, Any]]) -> List[ViolationAnalysis]:
        """Enhanced safety principle checking with context awareness"""
        violations = []
        
        # Truthfulness principle
        core_principles = self.framework.safety_principles.get('core_principles', [])
        if any('truthfulness' in principle.lower() for principle in core_principles):
            # Check for definitive claims about uncertain topics
            uncertain_claim_patterns = [
                r'\b(?:definitely|certainly|absolutely|guaranteed)\b.*?\b(?:will happen|future|prediction)\b',
                r'\b(?:i know for certain|it\'s a fact that|without a doubt)\b.*?\b(?:medical|scientific|financial)\b'
            ]
            
            for pattern in uncertain_claim_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(ViolationAnalysis(
                        violation_type=ViolationType.SAFETY_PRINCIPLE,
                        risk_level=RiskLevel.MEDIUM,
                        confidence=0.7,
                        matched_patterns=['truthfulness_violation'],
                        context_window=content[:150],
                        suggested_action="WARN"
                    ))
        
        # Helpfulness principle - check for refusal to help with legitimate requests
        if any('helpfulness' in principle.lower() for principle in core_principles):
            refusal_patterns = [
                r"i can't help with that",
                r"i won't assist with",
                r"i refuse to"
            ]
            
            # Only flag if the request seems legitimate (contextual analysis)
            if context and context.get('request_type') == 'legitimate':
                for pattern in refusal_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        violations.append(ViolationAnalysis(
                            violation_type=ViolationType.SAFETY_PRINCIPLE,
                            risk_level=RiskLevel.LOW,
                            confidence=0.6,
                            matched_patterns=['helpfulness_concern'],
                            context_window=content[:100],
                            suggested_action="REVIEW"
                        ))
        
        return violations
    
    def _check_context_violations(self, content: str, context: Optional[Dict[str, Any]]) -> List[ViolationAnalysis]:
        """Check for context-aware violations"""
        violations = []
        
        if not context:
            return violations
        
        # Check for context switching attacks
        if context.get('previous_topic') and context.get('current_topic'):
            prev_topic = context['previous_topic'].lower()
            curr_topic = context['current_topic'].lower()
            
            # Detect suspicious topic switches
            suspicious_switches = [
                ('safety', 'harmful'),
                ('educational', 'illegal'),
                ('creative', 'violence')
            ]
            
            for prev_pattern, curr_pattern in suspicious_switches:
                if prev_pattern in prev_topic and curr_pattern in curr_topic:
                    violations.append(ViolationAnalysis(
                        violation_type=ViolationType.CONTEXT_VIOLATION,
                        risk_level=RiskLevel.MEDIUM,
                        confidence=0.7,
                        matched_patterns=['suspicious_context_switch'],
                        context_window=f"Topic switch: {prev_topic} -> {curr_topic}",
                        suggested_action="REVIEW"
                    ))
        
        return violations
    
    def _detect_behavioral_anomalies(self, content: str) -> Optional[ViolationAnalysis]:
        """Detect behavioral anomalies in model output"""
        
        # Check for repetitive patterns (possible model breakdown)
        words = content.split()
        if len(words) > 10:
            # Check for excessive repetition
            word_counts = defaultdict(int)
            for word in words:
                word_counts[word] += 1
            
            max_repetition = max(word_counts.values())
            repetition_ratio = max_repetition / len(words)
            
            if repetition_ratio > 0.3:  # More than 30% repetition
                return ViolationAnalysis(
                    violation_type=ViolationType.BEHAVIORAL_ANOMALY,
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.8,
                    matched_patterns=['excessive_repetition'],
                    context_window=content[:100],
                    suggested_action="REGENERATE"
                )
        
        # Check for contradictory statements within short content
        if len(content) < 500:  # Only for shorter content
            contradiction_patterns = [
                (r'\byes\b.*?\bno\b', 'yes_no_contradiction'),
                (r'\btrue\b.*?\bfalse\b', 'true_false_contradiction'),
                (r'\bsafe\b.*?\bdangerous\b', 'safety_contradiction')
            ]
            
            for pattern, name in contradiction_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return ViolationAnalysis(
                        violation_type=ViolationType.BEHAVIORAL_ANOMALY,
                        risk_level=RiskLevel.LOW,
                        confidence=0.6,
                        matched_patterns=[name],
                        context_window=content[:150],
                        suggested_action="REVIEW"
                    )
        
        return None
    
    def _calculate_pattern_confidence(self, match: re.Match, content: str) -> float:
        """Calculate confidence score for pattern matches"""
        base_confidence = 0.8
        
        # Adjust based on match context
        match_text = match.group(0)
        match_length = len(match_text)
        content_length = len(content)
        
        # Longer matches in shorter content = higher confidence
        length_factor = min(1.0, match_length / 50.0)  # Cap at 50 chars
        context_factor = min(1.0, 100.0 / content_length) if content_length > 0 else 1.0
        
        confidence = base_confidence + (length_factor * 0.1) + (context_factor * 0.1)
        return min(1.0, confidence)
    
    def _extract_context_window(self, content: str, violations: List[ViolationAnalysis]) -> str:
        """Extract relevant context window around violations"""
        if not violations:
            return content[:100]
        
        # Find the position of the first violation
        first_violation_pattern = violations[0].matched_patterns[0] if violations[0].matched_patterns else None
        
        if first_violation_pattern and first_violation_pattern in self.pattern_cache:
            pattern = self.pattern_cache[first_violation_pattern]
            match = pattern.search(content)
            if match:
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                return content[start:end]
        
        return content[:150]
    
    def _determine_suggested_action(self, risk_level: RiskLevel, confidence: float) -> str:
        """Determine suggested action based on risk and confidence"""
        if risk_level == RiskLevel.CRITICAL:
            return "BLOCK"
        elif risk_level == RiskLevel.HIGH and confidence > 0.8:
            return "BLOCK"
        elif risk_level == RiskLevel.HIGH:
            return "WARN"
        elif risk_level == RiskLevel.MEDIUM and confidence > 0.7:
            return "WARN"
        elif risk_level == RiskLevel.MEDIUM:
            return "REVIEW"
        else:
            return "LOG"
    
    def _update_behavioral_metrics(self, content: str, analysis: Optional[ViolationAnalysis]) -> None:
        """Update behavioral metrics for pattern analysis"""
        content_type = 'violation' if analysis else 'clean'
        self.behavioral_metrics[content_type] += 1
        
        if analysis:
            risk_level = analysis.risk_level.value
            self.risk_scores[risk_level].append(analysis.confidence)
            
            # Keep only recent scores
            if len(self.risk_scores[risk_level]) > 100:
                self.risk_scores[risk_level] = self.risk_scores[risk_level][-50:]
    
    def _process_violations(self, analysis: ViolationAnalysis, content_sample: str) -> Tuple[bool, Optional[str], ViolationAnalysis]:
        """Process violations according to enforcement level"""
        self.violation_count += 1
        
        # Log violation
        violation_entry = {
            'timestamp': time.time(),
            'type': f'{analysis.violation_type.value}',
            'risk_level': analysis.risk_level.value,
            'confidence': analysis.confidence,
            'content_sample': content_sample,
            'patterns': analysis.matched_patterns,
            'action': self.framework.enforcement_level,
            'suggested_action': analysis.suggested_action
        }
        
        self.violation_log.append(violation_entry)
        
        # Determine response based on enforcement level and risk
        if self.framework.enforcement_level == "hard_fail":
            if analysis.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                return False, f"Constitutional violation: {', '.join(analysis.matched_patterns)}", analysis
            else:
                return True, f"Warning: {', '.join(analysis.matched_patterns)}", analysis
        elif self.framework.enforcement_level == "warn_and_proceed":
            return True, f"Warning: {', '.join(analysis.matched_patterns)}", analysis
        else:  # log_only
            return True, None, analysis
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get comprehensive violation summary with analytics"""
        with self._lock:
            recent_violations = list(self.violation_log)[-20:]  # Last 20
            
            # Calculate violation statistics
            violation_types = defaultdict(int)
            risk_levels = defaultdict(int)
            
            for violation in self.violation_log:
                violation_types[violation['type']] += 1
                risk_levels[violation['risk_level']] += 1
            
            # Performance metrics
            avg_analysis_time = (self.total_analysis_time / max(1, self.analysis_count)) * 1000  # ms
            
            # Behavioral insights
            behavioral_summary = dict(self.behavioral_metrics)
            risk_averages = {
                level: sum(scores) / len(scores) if scores else 0.0
                for level, scores in self.risk_scores.items()
            }
            
            return {
                'total_violations': self.violation_count,
                'recent_violations': recent_violations,
                'violation_types': dict(violation_types),
                'risk_level_distribution': dict(risk_levels),
                'performance_metrics': {
                    'total_analyses': self.analysis_count,
                    'avg_analysis_time_ms': round(avg_analysis_time, 2),
                    'patterns_loaded': len(self.pattern_cache)
                },
                'behavioral_metrics': behavioral_summary,
                'risk_score_averages': risk_averages,
                'active_patterns': len([p for p in self.safety_patterns if p.enabled])
            }
    
    def update_safety_patterns(self, new_patterns: List[SafetyPattern]) -> None:
        """Update safety patterns dynamically"""
        with self._lock:
            self.safety_patterns.extend(new_patterns)
            
            # Recompile pattern cache
            for pattern in new_patterns:
                if pattern.enabled:
                    try:
                        self.pattern_cache[pattern.name] = re.compile(
                            pattern.pattern, 
                            pattern.regex_flags
                        )
                    except re.error as e:
                        logger.warning(f"Invalid regex pattern '{pattern.name}': {e}")
    
    def disable_pattern(self, pattern_name: str) -> bool:
        """Disable a specific safety pattern"""
        with self._lock:
            for pattern in self.safety_patterns:
                if pattern.name == pattern_name:
                    pattern.enabled = False
                    if pattern_name in self.pattern_cache:
                        del self.pattern_cache[pattern_name]
                    return True
            return False
    
    def enable_pattern(self, pattern_name: str) -> bool:
        """Enable a specific safety pattern"""
        with self._lock:
            for pattern in self.safety_patterns:
                if pattern.name == pattern_name:
                    pattern.enabled = True
                    try:
                        self.pattern_cache[pattern.name] = re.compile(
                            pattern.pattern, 
                            pattern.regex_flags
                        )
                        return True
                    except re.error as e:
                        logger.warning(f"Invalid regex pattern '{pattern.name}': {e}")
                        return False
            return False


class HuggingFaceTranslator:
    """Translates CAS specifications to HuggingFace model configurations"""
    
    def __init__(self):
        self.config_templates = {
            'tokenizer': {
                'tokenizer_class': 'LlamaTokenizer',
                'use_fast': True,
                'padding_side': 'left',
                'truncation_side': 'left'
            },
            'generation': {
                'max_new_tokens': 2048,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'repetition_penalty': 1.1,
                'pad_token_id': 0,
                'eos_token_id': 2
            }
        }
    
    def translate(self, cas_spec: CASSpecification) -> Dict[str, Any]:
        """Translate CAS to HuggingFace configuration"""
        config = {
            'model_name': cas_spec.metadata.custom_name,
            'model_id': cas_spec.metadata.system_model_id,
            'description': cas_spec.metadata.description,
            'created_at': cas_spec.metadata.created_at,
            'cas_version': cas_spec.metadata.cas_version,
            'model_type': cas_spec.base_model.type,
            'architecture': cas_spec.base_model.architecture,
            'parameter_count': cas_spec.base_model.parameter_count
        }
        
        # Add tokenizer configuration
        config['tokenizer_config'] = self.config_templates['tokenizer'].copy()
        
        # Add generation configuration from cognitive profile
        default_profile = cas_spec.cognitive_architecture.get('default_profile', 'analytical')
        profiles = cas_spec.cognitive_architecture.get('profiles', {})
        
        if default_profile in profiles:
            profile_config = profiles[default_profile]
            param_prefs = profile_config.get('parameter_preferences', {})
            
            generation_config = self.config_templates['generation'].copy()
            generation_config.update({
                'temperature': param_prefs.get('temperature', 0.7),
                'top_p': param_prefs.get('top_p', 0.9),
                'top_k': param_prefs.get('top_k', 40),
                'max_new_tokens': cas_spec.base_model.context_length // 4
            })
            config['generation_config'] = generation_config
            
            # Add system prompt
            config['system_prompt'] = profile_config.get('system_prompt', '')
        
        # Add constitutional framework
        config['constitutional_framework'] = {
            'safety_mode': cas_spec.constitutional_framework.governor_mode,
            'enforcement_level': cas_spec.constitutional_framework.enforcement_level,
            'safety_principles': cas_spec.constitutional_framework.safety_principles
        }
        
        # Add memory configuration
        config['memory_config'] = {
            'max_memory_gb': cas_spec.memory_profile.max_memory_gb,
            'memory_management': cas_spec.memory_profile.memory_management,
            'tier_allocation': cas_spec.memory_profile.tier_allocation
        }
        
        return config
    
    def save_config(self, config: Dict[str, Any], output_path: Path) -> None:
        """Save HuggingFace configuration to JSON file"""
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


class LMStudioTranslator:
    """Translates CAS specifications to LM Studio configurations"""
    
    def __init__(self):
        """Initialize LM Studio translator with configuration validation."""
        self.supported_architectures = {'llama', 'gpt', 'mistral', 'falcon', 'mpt'}
        self.parameter_mappings = {
            'temperature': 'temperature',
            'top_p': 'top_p', 
            'top_k': 'top_k',
            'repeat_penalty': 'repeat_penalty',
            'max_tokens': 'max_new_tokens'
        }
        self.translation_cache = {}
        logger.debug("LM Studio translator initialized")
    
    def translate(self, cas_spec: CASSpecification) -> Dict[str, Any]:
        """Translate CAS to LM Studio configuration"""
        config = {
            'name': cas_spec.metadata.custom_name,
            'description': cas_spec.metadata.description,
            'model': {
                'path': cas_spec.base_model.path,
                'type': cas_spec.base_model.type,
                'architecture': cas_spec.base_model.architecture,
                'parameters': cas_spec.base_model.parameter_count,
                'context_length': cas_spec.base_model.context_length,
                'quantization': cas_spec.base_model.quantization
            },
            'inference': {
                'gpu_layers': -1,  # Use all GPU layers
                'threads': -1,     # Auto-detect threads
                'batch_size': 512,
                'rope_frequency_base': 10000,
                'rope_frequency_scale': 1.0
            },
            'safety': {
                'mode': cas_spec.constitutional_framework.governor_mode,
                'enforcement': cas_spec.constitutional_framework.enforcement_level,
                'content_filters': cas_spec.constitutional_framework.content_filters
            }
        }
        
        # Add cognitive profile settings
        default_profile = cas_spec.cognitive_architecture.get('default_profile', 'analytical')
        profiles = cas_spec.cognitive_architecture.get('profiles', {})
        
        if default_profile in profiles:
            profile_config = profiles[default_profile]
            param_prefs = profile_config.get('parameter_preferences', {})
            
            config['sampling'] = {
                'temperature': param_prefs.get('temperature', 0.7),
                'top_p': param_prefs.get('top_p', 0.9),
                'top_k': param_prefs.get('top_k', 40),
                'min_p': param_prefs.get('min_p', 0.05),
                'typical_p': param_prefs.get('typical_p', 1.0),
                'repeat_penalty': param_prefs.get('repeat_penalty', 1.1),
                'repeat_last_n': param_prefs.get('repeat_last_n', 64),
                'penalize_nl': param_prefs.get('penalize_nl', True)
            }
            
            config['chat_template'] = profile_config.get('system_prompt', '')
        
        # Add memory management
        config['memory'] = {
            'max_ram_gb': cas_spec.memory_profile.max_memory_gb,
            'cache_type': cas_spec.memory_profile.memory_management,
            'offload_kqv': True,
            'flash_attention': True
        }
        
        return config
    
    def save_config(self, config: Dict[str, Any], output_path: Path) -> None:
        """Save LM Studio configuration to JSON file"""
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


class OllamaTranslator:
    """Translates CAS specifications to Ollama Modelfiles"""
    
    def __init__(self):
        """Initialize Ollama translator with Modelfile template handling."""
        self.supported_parameters = {
            'temperature', 'top_p', 'top_k', 'repeat_penalty', 
            'num_ctx', 'num_predict', 'stop'
        }
        self.base_model_mappings = {
            'llama3.1': 'llama3.1:8b-instruct',
            'llama3': 'llama3:8b-instruct', 
            'mistral': 'mistral:7b-instruct',
            'deepseek': 'deepseek-r1:7b-instruct'
        }
        self.template_cache = {}
        logger.debug("Ollama translator initialized")
    
    def translate(self, cas_spec: CASSpecification) -> str:
        """Translate CAS to Ollama Modelfile format"""
        lines = []
        
        # Header comment
        lines.append("# Generated from Cognitive Architecture Specification (CAS)")
        lines.append(f"# Model: {cas_spec.metadata.custom_name}")
        lines.append(f"# Created: {cas_spec.metadata.created_at}")
        lines.append(f"# CAS Version: {cas_spec.metadata.cas_version}")
        lines.append("")
        
        # Base model
        base_model_name = self._map_base_model(cas_spec.base_model.path)
        lines.append(f"FROM {base_model_name}")
        lines.append("")
        
        # Template format
        ollama_config = cas_spec.platform_translation.ollama
        template_format = ollama_config.get('template_format', 'chat')
        
        if template_format == 'chat':
            lines.append('TEMPLATE """<|begin_of_text|>{{- if .System }}<|start_header_id|>system<|end_header_id|>')
            lines.append('')
            lines.append('{{ .System }}<|eot_id|>')
            lines.append('{{- end }}')
            lines.append('{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>')
            lines.append('')
            lines.append('{{ .Content }}<|eot_id|>')
            lines.append('{{- end }}<|start_header_id|>assistant<|end_header_id|>')
            lines.append('')
            lines.append('"""')
            lines.append('')
        
        # Parameters
        default_profile = cas_spec.cognitive_architecture.get('default_profile', 'analytical')
        profiles = cas_spec.cognitive_architecture.get('profiles', {})
        
        if default_profile in profiles:
            profile_config = profiles[default_profile]
            param_prefs = profile_config.get('parameter_preferences', {})
            
            lines.append(f"PARAMETER temperature {param_prefs.get('temperature', 0.7)}")
            lines.append(f"PARAMETER top_p {param_prefs.get('top_p', 0.9)}")
            lines.append(f"PARAMETER top_k {param_prefs.get('top_k', 40)}")
            lines.append("PARAMETER repeat_penalty 1.1")
            lines.append(f"PARAMETER num_ctx {cas_spec.base_model.context_length}")
            lines.append("")
        
        # System prompt
        if default_profile in profiles:
            profile_config = profiles[default_profile]
            system_prompt = profile_config.get('system_prompt', '')
            
            # Add constitutional principles
            safety_principles = cas_spec.constitutional_framework.safety_principles
            if safety_principles:
                system_prompt += "\n\nConstitutional Principles:\n"
                for category, principles in safety_principles.items():
                    for principle in principles:
                        system_prompt += f"- {principle}\n"
            
            lines.append(f'SYSTEM """{system_prompt}"""')
            lines.append("")
        
        # Stop sequences
        lines.append('PARAMETER stop "<|start_header_id|>"')
        lines.append('PARAMETER stop "<|end_header_id|>"')
        lines.append('PARAMETER stop "<|eot_id|>"')
        lines.append('PARAMETER stop "<|begin_of_text|>"')
        
        return '\n'.join(lines)
    
    def _map_base_model(self, model_path: str) -> str:
        """Map model path to Ollama model name"""
        # Simple mapping - could be more sophisticated
        if 'deepseek' in model_path.lower():
            return 'deepseek-r1:7b-instruct'
        elif 'llama' in model_path.lower():
            return 'llama3.1:8b-instruct'
        elif 'mistral' in model_path.lower():
            return 'mistral:7b-instruct'
        else:
            return 'llama3.1:8b-instruct'  # Default fallback


# Integration class for model_creation.py compatibility
class CASModelCreationIntegration:
    """Integrates CAS system with existing model_creation.py workflow"""
    
    def __init__(self, base_directory: Optional[Path] = None):
        self.base_directory = Path(base_directory or "./model_parameters")
        self.cas_generator = CASGenerator()
        self.cas_parser = CASParser()
        
        # Initialize all translators
        self.ollama_translator = OllamaTranslator()
        self.huggingface_translator = HuggingFaceTranslator()
        self.lmstudio_translator = LMStudioTranslator()
        
        # Setup directory structure
        self.cas_dir = self.base_directory / "cas_files"
        self.ollama_dir = self.base_directory / "ollama_exports"
        self.huggingface_dir = self.base_directory / "huggingface_exports"
        self.lmstudio_dir = self.base_directory / "lmstudio_exports"
        
        # Create all directories
        for directory in [self.cas_dir, self.ollama_dir, self.huggingface_dir, self.lmstudio_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def create_cas_model(self, 
                        model_name: str,
                        model_path: str,
                        model_type: str = "gguf",
                        cognitive_profile: str = "analytical",
                        safety_mode: str = "balanced",
                        **kwargs) -> Tuple[Path, Dict[str, Any]]:
        """Create a new CAS model configuration"""
        
        # Generate CAS content
        cas_content, template_vars = self.cas_generator.generate_from_template(
            template_name="default",
            model_name=model_name,
            model_path=model_path,
            model_type=model_type,
            cognitive_profile=cognitive_profile,
            safety_mode=safety_mode,
            **kwargs
        )
        
        # Save CAS file
        cas_filename = f"{model_name.lower().replace(' ', '_')}.cas.yml"
        cas_path = self.cas_dir / cas_filename
        self.cas_generator.save_cas_file(cas_content, cas_path)
        
        # Parse and validate
        cas_spec, errors, warnings = self.cas_parser.parse_file(cas_path)
        
        if errors:
            logger.error(f"CAS validation errors: {errors}")
            raise ValueError(f"Invalid CAS specification: {errors}")
        
        if warnings:
            logger.warning(f"CAS validation warnings: {warnings}")
        
        # Generate exports for all platforms
        export_results = self._generate_all_exports(cas_spec, model_name)
        
        # Calculate file hashes
        cas_hash = hashlib.sha256(cas_content.encode('utf-8')).hexdigest()
        
        # Return paths and metadata
        result_metadata = {
            'cas_path': cas_path,
            'cas_hash': cas_hash,
            'system_model_id': template_vars['system_model_id'],
            'template_vars': template_vars,
            'validation_errors': errors,
            'validation_warnings': warnings,
            'exports': export_results
        }
        
        logger.info(f"Created CAS model: {model_name}")
        logger.info(f"  CAS file: {cas_path}")
        for platform, export_info in export_results.items():
            logger.info(f"  {platform.title()} export: {export_info['path']}")
        
        return cas_path, result_metadata
    
    def _generate_all_exports(self, cas_spec: CASSpecification, model_name: str) -> Dict[str, Dict[str, Any]]:
        """Generate exports for all supported platforms"""
        exports = {}
        model_filename = model_name.lower().replace(' ', '_')
        
        try:
            # Ollama export
            ollama_content = self.ollama_translator.translate(cas_spec)
            ollama_path = self.ollama_dir / f"{model_filename}.modelfile"
            with open(ollama_path, 'w', encoding='utf-8') as f:
                f.write(ollama_content)
            
            exports['ollama'] = {
                'path': ollama_path,
                'hash': hashlib.sha256(ollama_content.encode('utf-8')).hexdigest(),
                'format': 'modelfile',
                'size_bytes': len(ollama_content)
            }
        except Exception as e:
            logger.error(f"Failed to generate Ollama export: {e}")
            exports['ollama'] = {'error': str(e)}
        
        try:
            # HuggingFace export
            hf_config = self.huggingface_translator.translate(cas_spec)
            hf_path = self.huggingface_dir / f"{model_filename}_hf_config.json"
            self.huggingface_translator.save_config(hf_config, hf_path)
            
            with open(hf_path, 'r', encoding='utf-8') as f:
                hf_content = f.read()
            
            exports['huggingface'] = {
                'path': hf_path,
                'hash': hashlib.sha256(hf_content.encode('utf-8')).hexdigest(),
                'format': 'json',
                'size_bytes': len(hf_content)
            }
        except Exception as e:
            logger.error(f"Failed to generate HuggingFace export: {e}")
            exports['huggingface'] = {'error': str(e)}
        
        try:
            # LM Studio export
            lms_config = self.lmstudio_translator.translate(cas_spec)
            lms_path = self.lmstudio_dir / f"{model_filename}_lms_config.json"
            self.lmstudio_translator.save_config(lms_config, lms_path)
            
            with open(lms_path, 'r', encoding='utf-8') as f:
                lms_content = f.read()
            
            exports['lmstudio'] = {
                'path': lms_path,
                'hash': hashlib.sha256(lms_content.encode('utf-8')).hexdigest(),
                'format': 'json',
                'size_bytes': len(lms_content)
            }
        except Exception as e:
            logger.error(f"Failed to generate LM Studio export: {e}")
            exports['lmstudio'] = {'error': str(e)}
        
        return exports
    
    def load_cas_model(self, cas_path: Union[str, Path]) -> Tuple[CASSpecification, ConstitutionalGovernor]:
        """Load CAS model and create constitutional governor"""
        
        cas_spec, errors, warnings = self.cas_parser.parse_file(cas_path)
        
        if errors:
            raise ValueError(f"Failed to load CAS model: {errors}")
        
        if warnings:
            logger.warning(f"CAS warnings: {warnings}")
        
        # Create constitutional governor
        governor = ConstitutionalGovernor(cas_spec.constitutional_framework)
        
        return cas_spec, governor
    
    def export_to_platform(self, cas_path: Union[str, Path], platform: str) -> Path:
        """Export CAS model to specified platform"""
        cas_spec, errors, warnings = self.cas_parser.parse_file(cas_path)
        
        if errors:
            raise ValueError(f"Cannot export invalid CAS: {errors}")
        
        cas_path = Path(cas_path)
        base_filename = cas_path.stem.replace('.cas', '')
        
        if platform.lower() == 'ollama':
            content = self.ollama_translator.translate(cas_spec)
            output_path = self.ollama_dir / f"{base_filename}.modelfile"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        elif platform.lower() == 'huggingface':
            config = self.huggingface_translator.translate(cas_spec)
            output_path = self.huggingface_dir / f"{base_filename}_hf_config.json"
            self.huggingface_translator.save_config(config, output_path)
            
        elif platform.lower() == 'lmstudio':
            config = self.lmstudio_translator.translate(cas_spec)
            output_path = self.lmstudio_dir / f"{base_filename}_lms_config.json"
            self.lmstudio_translator.save_config(config, output_path)
            
        else:
            raise ValueError(f"Unsupported platform: {platform}")
        
        logger.info(f"Exported to {platform}: {output_path}")
        return output_path
    
    def export_to_ollama(self, cas_path: Union[str, Path]) -> Path:
        """Export CAS model to Ollama Modelfile"""
        return self.export_to_platform(cas_path, 'ollama')
    
    def export_to_huggingface(self, cas_path: Union[str, Path]) -> Path:
        """Export CAS model to HuggingFace configuration"""
        return self.export_to_platform(cas_path, 'huggingface')
    
    def export_to_lmstudio(self, cas_path: Union[str, Path]) -> Path:
        """Export CAS model to LM Studio configuration"""
        return self.export_to_platform(cas_path, 'lmstudio')
    
    def export_all_platforms(self, cas_path: Union[str, Path]) -> Dict[str, Path]:
        """Export CAS model to all supported platforms"""
        results = {}
        
        for platform in ['ollama', 'huggingface', 'lmstudio']:
            try:
                results[platform] = self.export_to_platform(cas_path, platform)
            except Exception as e:
                logger.error(f"Failed to export to {platform}: {e}")
                results[platform] = None
        
        return results
    
    def import_from_ollama(self, modelfile_path: Union[str, Path], model_name: str) -> Tuple[Path, Dict[str, Any]]:
        """Import Ollama Modelfile and convert to CAS (bidirectional conversion)"""
        modelfile_path = Path(modelfile_path)
        
        if not modelfile_path.exists():
            raise FileNotFoundError(f"Modelfile not found: {modelfile_path}")
        
        # Parse Ollama Modelfile
        with open(modelfile_path, 'r', encoding='utf-8') as f:
            modelfile_content = f.read()
        
        # Extract model configuration from Modelfile
        extracted_config = self._parse_ollama_modelfile(modelfile_content)
        
        # Create CAS model from extracted configuration
        return self.create_cas_model(
            model_name=model_name,
            model_path=extracted_config.get('base_model', './models/default.gguf'),
            model_type=extracted_config.get('model_type', 'gguf'),
            cognitive_profile=extracted_config.get('cognitive_profile', 'analytical'),
            safety_mode=extracted_config.get('safety_mode', 'balanced'),
            description=f"Imported from Ollama Modelfile: {modelfile_path.name}",
            **extracted_config.get('additional_params', {})
        )
    
    def _parse_ollama_modelfile(self, content: str) -> Dict[str, Any]:
        """Parse Ollama Modelfile content and extract configuration"""
        config = {
            'base_model': './models/default.gguf',
            'model_type': 'gguf',
            'cognitive_profile': 'analytical',
            'safety_mode': 'balanced',
            'additional_params': {}
        }
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract FROM directive
            if line.startswith('FROM '):
                base_model = line[5:].strip()
                config['base_model'] = base_model
            
            # Extract PARAMETER directives
            elif line.startswith('PARAMETER '):
                param_line = line[10:].strip()
                parts = param_line.split(' ', 1)
                if len(parts) == 2:
                    param_name, param_value = parts
                    
                    # Convert numeric values
                    try:
                        if '.' in param_value:
                            param_value = float(param_value)
                        else:
                            param_value = int(param_value)
                    except ValueError:
                        pass  # Keep as string
                    
                    config['additional_params'][param_name] = param_value
            
            # Extract SYSTEM directive
            elif line.startswith('SYSTEM '):
                system_prompt = line[7:].strip().strip('"""')
                config['additional_params']['system_prompt'] = system_prompt
        
        return config


# Example usage and testing
if __name__ == "__main__":
    # Test CAS system
    integration = CASModelCreationIntegration()
    
    try:
        # Create a new CAS model
        cas_path, metadata = integration.create_cas_model(
            model_name="Revolutionary Test Model",
            model_path="./models/deepseek-r1-7b.gguf",
            model_type="gguf",
            cognitive_profile="analytical",
            safety_mode="balanced",
            description="Test model for CAS system validation"
        )
        
        logger.info(f"Created CAS model: {cas_path}")
        logger.info(f"System Model ID: {metadata['system_model_id']}")
        logger.info(f"CAS Hash: {metadata['cas_hash'][:16]}...")
        
        # Load the model
        cas_spec, governor = integration.load_cas_model(cas_path)
        logger.info(f"Loaded model: {cas_spec.metadata.custom_name}")
        
        # Test constitutional governor
        is_safe, warning, analysis = governor.check_input("How do I analyze complex systems?")
        logger.info(f"Input check: {'Safe' if is_safe else 'Blocked'}")
        if warning:
            logger.warning(f"Safety warning: {warning}")
        if analysis:
            logger.debug(f"Analysis details: {analysis.summary}")
        
        logger.info("CAS system test completed successfully!")
        
    except Exception as e:
        logger.error(f"CAS system test failed: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
