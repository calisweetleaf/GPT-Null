"""
CAS Integration Bridge - Enhanced model_creation.py Compatibility
================================================================

Bridges the revolutionary CAS system with existing model_creation.py workflow.
Extends your current CustomModelType, ModelFileTemplate, and reporting systems
with CAS capabilities while maintaining backward compatibility.

Key Features:
- Extends existing CustomModelType enum with CAS support
- Enhances ModelFileTemplate to include CAS templates
- Integrates CAS generation into existing workflow
- Maintains SHA-256 tracking and versioning
- Extends ModelConfigReport with CAS metadata
- Provides seamless upgrade path for existing projects

Author: Cybernetic Architecture Division  
License: MIT
"""

import os
import logging
import uuid
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, cast
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# Import existing model_creation components
from model_creation import (
    CustomModelType as BaseCustomModelType,
    CustomModelName as BaseCustomModelName,
    CustomModelFile,
    ModelFileTemplate as BaseModelFileTemplate,
    ModelConfigReport as BaseModelConfigReport,
    GenerateModelFile as BaseGenerateModelFile,
    ModelFileDirectory,
    compute_sha256_file
)

# Configure logging
logger = logging.getLogger(__name__)

# Import CAS system components
from cas_system import (
    CASSpecification, CASGenerator, CASParser, 
    OllamaTranslator, CASModelCreationIntegration, SafetyMode
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedCustomModelType(Enum):
    """Enhanced CustomModelType with CAS support - extends the base enum"""
    # Import all original types from base enum
    OLLAMA = BaseCustomModelType.OLLAMA.value
    MSTY = BaseCustomModelType.MSTY.value
    LMSTUDIO = BaseCustomModelType.LMSTUDIO.value
    OPEN_ROUTER = BaseCustomModelType.OPEN_ROUTER.value
    OPEN_ROUTER_MODELNAME = BaseCustomModelType.OPEN_ROUTER_MODELNAME.value
    LOCAL_PYTHON = BaseCustomModelType.LOCAL_PYTHON.value
    HUGGINGFACE = BaseCustomModelType.HUGGINGFACE.value
    GGUF = BaseCustomModelType.GGUF.value
    GGML = BaseCustomModelType.GGML.value
    PyTorch = BaseCustomModelType.PyTorch.value
    TensorFlow = BaseCustomModelType.TensorFlow.value
    JAX = BaseCustomModelType.JAX.value
    ONNX = BaseCustomModelType.ONNX.value
    CUSTOM_NEURAL_NETWORKS = BaseCustomModelType.CUSTOM_NEURAL_NETWORKS.value
    ALGO = BaseCustomModelType.ALGO.value
    DIFFUSION = BaseCustomModelType.DIFFUSION.value
    GENERATIVE = BaseCustomModelType.GENERATIVE.value
    EMBEDDING = BaseCustomModelType.EMBEDDING.value
    CUSTOM_UNIVERSAL_MODEL = BaseCustomModelType.CUSTOM_UNIVERSAL_MODEL.value
    
    # New CAS-enhanced types
    CAS_OLLAMA = "cas_ollama"
    CAS_GGUF = "cas_gguf"
    CAS_PYTORCH = "cas_pytorch"
    CAS_UNIVERSAL = "cas_universal"
    CAS_HYBRID = "cas_hybrid"
    
    @classmethod
    def from_base_type(cls, base_type: BaseCustomModelType) -> 'EnhancedCustomModelType':
        """Convert base CustomModelType to enhanced version"""
        for enhanced_type in cls:
            if enhanced_type.value == base_type.value:
                return enhanced_type
        return cls.CUSTOM_UNIVERSAL_MODEL


@dataclass
class EnhancedCustomModelName(BaseCustomModelName):
    """Enhanced CustomModelName with CAS metadata - inherits from base class"""
    # CAS-specific fields
    cognitive_profile: Optional[str] = None
    safety_mode: Optional[str] = None
    constitutional_principles: List[str] = field(default_factory=list)
    reasoning_framework: Optional[str] = None
    
    # Store enhanced type separately to avoid type conflicts
    enhanced_model_type: Optional[EnhancedCustomModelType] = None
    
    def __post_init__(self):
        """Store enhanced version of model_type separately without modifying the base attribute"""
        if isinstance(self.model_type, BaseCustomModelType):
            # Store enhanced type separately but don't modify base class attribute
            self.enhanced_model_type = EnhancedCustomModelType.from_base_type(self.model_type)
        elif isinstance(self.model_type, str):
            # Try to convert string to enhanced type
            try:
                self.enhanced_model_type = EnhancedCustomModelType(self.model_type)
            except ValueError:
                self.enhanced_model_type = EnhancedCustomModelType.CUSTOM_UNIVERSAL_MODEL
        elif isinstance(self.model_type, EnhancedCustomModelType):
            self.enhanced_model_type = self.model_type
    
    def is_cas_enabled(self) -> bool:
        """Check if this model uses CAS features"""
        if self.enhanced_model_type:
            return self.enhanced_model_type.value.startswith('cas_')
        return False
    
    def to_base_model_name(self) -> BaseCustomModelName:
        """Convert to base CustomModelName for compatibility"""
        base_type = BaseCustomModelType.CUSTOM_UNIVERSAL_MODEL
        try:
            if self.enhanced_model_type:
                base_type = BaseCustomModelType(self.enhanced_model_type.value)
        except ValueError as e:
            logger.warning(f"Failed to convert enhanced model type {self.enhanced_model_type.value} to base type, using default: {e}")
            base_type = BaseCustomModelType.CUSTOM_UNIVERSAL_MODEL
        
        return BaseCustomModelName(
            name=self.name,
            model_type=base_type,
            version=self.version,
            description=self.description,
            created_at=self.created_at,
            system_model_id=self.system_model_id
        )


@dataclass
class EnhancedModelFileTemplate(BaseModelFileTemplate):
    """Enhanced ModelFileTemplate with CAS support - inherits from base class"""
    # CAS-specific fields
    cas_enabled: bool = False
    cas_template_path: Optional[str] = None
    constitutional_template: Optional[str] = None
    cognitive_profiles: List[str] = field(default_factory=list)
    reasoning_frameworks: List[str] = field(default_factory=list)
    
    # Store enhanced type separately to avoid type conflicts
    enhanced_target_type: Optional[EnhancedCustomModelType] = None
    
    def __post_init__(self):
        """Store enhanced version of target_type separately without modifying the base attribute"""
        if isinstance(self.target_type, BaseCustomModelType):
            self.enhanced_target_type = EnhancedCustomModelType.from_base_type(self.target_type)
        elif isinstance(self.target_type, str):
            try:
                self.enhanced_target_type = EnhancedCustomModelType(self.target_type)
            except ValueError:
                self.enhanced_target_type = EnhancedCustomModelType.CUSTOM_UNIVERSAL_MODEL
        elif isinstance(self.target_type, EnhancedCustomModelType):
            self.enhanced_target_type = self.target_type
    
    def supports_cas(self) -> bool:
        """Check if template supports CAS features"""
        if self.cas_enabled:
            return True
        if self.enhanced_target_type:
            return self.enhanced_target_type.value.startswith('cas_')
        return False
    
    def to_base_template(self) -> BaseModelFileTemplate:
        """Convert to base ModelFileTemplate for compatibility"""
        base_type = BaseCustomModelType.CUSTOM_UNIVERSAL_MODEL
        try:
            if self.enhanced_target_type:
                base_type = BaseCustomModelType(self.enhanced_target_type.value)
        except ValueError as e:
            logger.warning(f"Failed to convert enhanced target type {self.enhanced_target_type.value} to base type, using default: {e}")
            base_type = BaseCustomModelType.CUSTOM_UNIVERSAL_MODEL
        
        return BaseModelFileTemplate(
            template_id=self.template_id,
            name=self.name,
            content_template=self.content_template,
            required_variables=self.required_variables,
            optional_variables=self.optional_variables,
            target_type=base_type,
            description=self.description
        )


@dataclass
class EnhancedModelConfigReport(BaseModelConfigReport):
    """Enhanced ModelConfigReport with CAS integration - inherits from base class"""
    # CAS-specific fields
    cas_specification: Optional[Dict[str, Any]] = None
    constitutional_framework: Optional[Dict[str, Any]] = None
    cognitive_architecture: Optional[Dict[str, Any]] = None
    memory_profile: Optional[Dict[str, Any]] = None
    adaptation_metrics: Optional[Dict[str, Any]] = None
    ollama_export_path: Optional[str] = None
    
    # Store the enhanced model type separately to maintain type safety
    enhanced_model_type: Optional[EnhancedCustomModelType] = None
    
    def __post_init__(self):
        """Convert model_type to enhanced version if needed"""
        # Store the enhanced version separately
        if isinstance(self.model_type, BaseCustomModelType):
            self.enhanced_model_type = EnhancedCustomModelType.from_base_type(self.model_type)
        elif isinstance(self.model_type, str):
            try:
                self.enhanced_model_type = EnhancedCustomModelType(self.model_type)
            except ValueError:
                self.enhanced_model_type = EnhancedCustomModelType.CUSTOM_UNIVERSAL_MODEL
        elif isinstance(self.model_type, EnhancedCustomModelType):
            self.enhanced_model_type = self.model_type
    
    def has_cas_data(self) -> bool:
        """Check if report contains CAS data"""
        return self.cas_specification is not None
    
    def get_enhanced_model_type(self) -> EnhancedCustomModelType:
        """Get the enhanced model type"""
        if self.enhanced_model_type is None:
            return EnhancedCustomModelType.from_base_type(self.model_type) if isinstance(self.model_type, BaseCustomModelType) else EnhancedCustomModelType.CUSTOM_UNIVERSAL_MODEL
        return self.enhanced_model_type


class EnhancedGenerateModelFile(BaseGenerateModelFile):
    """Enhanced model file generator with CAS integration - inherits from base class"""
    
    def __init__(self, base_directory: Optional[Union[str, Path]] = None):
        super().__init__(base_directory)
        
        # Initialize CAS components
        self.cas_integration = CASModelCreationIntegration(self.base_directory)
        self.cas_generator = CASGenerator()
        self.cas_parser = CASParser()
        self.ollama_translator = OllamaTranslator()
        
        # Setup enhanced directory structure
        self.setup_enhanced_directory_structure()
    
    def setup_enhanced_directory_structure(self) -> None:
        """Sets up the enhanced directory structure for CAS-enabled models.

        This method first calls the base class's directory setup to ensure core directories
        are in place, then creates additional CAS-specific directories for specifications,
        templates, Ollama exports, constitutional configurations, cognitive profiles,
        and adaptation logs.
        """
        super().setup_model_file_directory()
        
        # Additional CAS directories not covered by the base setup
        cas_additional_dirs = [
            self.base_directory / "cas_specifications",
            self.base_directory / "cas_templates", 
            self.base_directory / "ollama_exports",
            self.base_directory / "constitutional_configs",
            self.base_directory / "cognitive_profiles",
            self.base_directory / "adaptation_logs"
        ]
        
        for directory in cas_additional_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created enhanced directory: {directory}")
    
    def create_enhanced_model_name(self, 
                                 name: str,
                                 model_type: Union[str, EnhancedCustomModelType, BaseCustomModelType],
                                 version: str = "1.0.0",
                                 description: str = "",
                                 cognitive_profile: str = "analytical",
                                 safety_mode: str = "balanced",
                                 **kwargs) -> EnhancedCustomModelName:
        """Creates an EnhancedCustomModelName object with CAS-specific metadata."""
        
        # Convert various model type inputs to EnhancedCustomModelType
        enhanced_type = None
        base_type = BaseCustomModelType.CUSTOM_UNIVERSAL_MODEL
        
        if isinstance(model_type, BaseCustomModelType):
            enhanced_type = EnhancedCustomModelType.from_base_type(model_type)
            base_type = model_type
        elif isinstance(model_type, str):
            try:
                enhanced_type = EnhancedCustomModelType(model_type)
                # Try to convert to base type too for compatibility
                try:
                    base_type = BaseCustomModelType(model_type)
                except ValueError as e:
                    logger.debug(f"Could not convert model type to base type: {e}")
                    base_type = None
            except ValueError:
                enhanced_type = EnhancedCustomModelType.CAS_UNIVERSAL
        elif isinstance(model_type, EnhancedCustomModelType):
            enhanced_type = model_type
            # Try to convert to base type for compatibility
            try:
                base_type = BaseCustomModelType(model_type.value)
            except ValueError as e:
                logger.debug(f"Could not convert enhanced model type to base type: {e}")
                base_type = BaseCustomModelType.CUSTOM_UNIVERSAL_MODEL
        
        # Create model with proper base type but store enhanced type separately
        model = EnhancedCustomModelName(
            name=name,
            model_type=base_type,
            version=version,
            description=description,
            cognitive_profile=cognitive_profile,
            safety_mode=safety_mode,
            reasoning_framework=kwargs.get('reasoning_framework'),
            constitutional_principles=kwargs.get('constitutional_principles', []),
            enhanced_model_type=enhanced_type
        )
        
        return model
    
    async def generate_cas_enhanced_model(self,
                                        name: str,
                                        model_path: str,
                                        model_type: str = "gguf",
                                        cognitive_profile: str = "analytical",
                                        safety_mode: str = "balanced",
                                        export_ollama: bool = True,
                                        **kwargs) -> Tuple[Dict[str, Path], EnhancedModelConfigReport]:
        """Generate CAS-enhanced model with full integration"""
        
        # Create enhanced model name
        enhanced_model_name = self.create_enhanced_model_name(
            name=name,
            model_type=f"cas_{model_type}",
            cognitive_profile=cognitive_profile,
            safety_mode=safety_mode,
            **kwargs
        )
        
        # Generate CAS specification
        cas_path, cas_metadata = self.cas_integration.create_cas_model(
            model_name=name,
            model_path=model_path,
            model_type=model_type,
            cognitive_profile=cognitive_profile,
            safety_mode=safety_mode,
            **kwargs
        )
        
        # Load and validate CAS specification
        cas_spec, constitutional_governor = self.cas_integration.load_cas_model(cas_path)
        
        # Generate traditional modelfile for compatibility
        traditional_modelfile = await self._generate_traditional_modelfile(
            enhanced_model_name, cas_spec
        )
        
        # Save traditional modelfile
        traditional_path = self.directory_manager.get_model_file_path(
            enhanced_model_name.to_filename(), "modelfile"
        )
        with open(traditional_path, 'w', encoding='utf-8') as f:
            f.write(traditional_modelfile.content)
        
        # Export to Ollama if requested
        ollama_path = None
        if export_ollama:
            ollama_path = self.cas_integration.export_to_ollama(cas_path)
        
        # Generate parameter report
        parameters_report = self._generate_enhanced_parameters_report(
            enhanced_model_name, cas_spec, model_path
        )
        
        # Save parameter report
        parameters_path = self.directory_manager.get_model_file_path(
            f"{enhanced_model_name.to_filename()}_parameters", "txt"
        )
        with open(parameters_path, 'w', encoding='utf-8') as f:
            f.write(parameters_report)
        
        # Calculate all file hashes
        file_hashes = {
            'cas_specification': self._calculate_file_hash(cas_path),
            'traditional_modelfile': self._calculate_file_hash(traditional_path),
            'parameters_report': self._calculate_file_hash(parameters_path)
        }
        
        if ollama_path:
            file_hashes['ollama_export'] = self._calculate_file_hash(ollama_path)

        # Create enhanced configuration report
        config_report = EnhancedModelConfigReport(
            report_id=str(uuid.uuid4()),
            model_name=enhanced_model_name.name,
            model_type=enhanced_model_name.model_type,
            configuration={
                'model_path': model_path,
                'cognitive_profile': cognitive_profile,
                'safety_mode': safety_mode,
                'cas_enabled': True,
                **kwargs
            },
            system_model_id=enhanced_model_name.system_model_id,
            artifacts=file_hashes,
            cas_specification=self._cas_spec_to_dict(cas_spec),
            constitutional_framework=self._constitutional_framework_to_dict(cas_spec),
            cognitive_architecture=cas_spec.cognitive_architecture,
            memory_profile=self._memory_profile_to_dict(cas_spec),
            ollama_export_path=str(ollama_path) if ollama_path else None
        )
        
        # Validate configuration
        config_report.errors, config_report.warnings = self._validate_enhanced_configuration(
            config_report.configuration, enhanced_model_name.model_type
        )
        config_report.validation_status = "passed" if not config_report.errors else "failed"
        
        # Save configuration report
        report_path = self._save_enhanced_config_report(config_report)
        
        # Collect all generated paths
        generated_paths = {
            'cas_specification': cas_path,
            'traditional_modelfile': traditional_path,
            'parameters_report': parameters_path,
            'config_report': report_path
        }
        
        if ollama_path:
            generated_paths['ollama_export'] = ollama_path
        
        logger.info(f"Generated enhanced CAS model: {name}")
        logger.info(f"  System Model ID: {enhanced_model_name.system_model_id}")
        logger.info(f"  CAS Specification: {cas_path}")
        logger.info(f"  Traditional Modelfile: {traditional_path}")
        if ollama_path:
            logger.info(f"  Ollama Export: {ollama_path}")
        
        return generated_paths, config_report

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculates the SHA256 hash of a given file."""
        return compute_sha256_file(file_path)
    
    async def _generate_traditional_modelfile(self, 
                                            model_name: EnhancedCustomModelName,
                                            cas_spec: CASSpecification) -> 'CustomModelFile':
        """Generate traditional modelfile from CAS specification for compatibility"""
        
        # Extract default cognitive profile
        default_profile = cas_spec.cognitive_architecture.get('default_profile', 'analytical')
        profiles = cas_spec.cognitive_architecture.get('profiles', {})
        
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
        else:
            system_prompt = "You are a helpful AI assistant."
        
        # Create traditional modelfile content
        content = f"""# Enhanced Model: {model_name.name}
# Generated from CAS Specification
# System Model ID: {model_name.system_model_id}
# Cognitive Profile: {model_name.cognitive_profile}
# Safety Mode: {model_name.safety_mode}

FROM {cas_spec.base_model.path}

TEMPLATE \"\"\"<|begin_of_text|>{{{{- if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>
{{{{- end }}}}
{{{{- range .Messages }}}}<|start_header_id|>{{{{ .Role }}}}<|end_header_id|>

{{{{ .Content }}}}<|eot_id|>
{{{{- end }}}}<|start_header_id|>assistant<|end_header_id|>

\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx {cas_spec.base_model.context_length}

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|begin_of_text|>"
"""
        
        # Return a proper CustomModelFile instance
        return CustomModelFile(
            name=model_name.to_filename(),
            content=content,
            file_type='modelfile',
            metadata={
                'source': 'cas_enhanced',
                'cas_enabled': True,
                'system_model_id': model_name.system_model_id
            }
        )
    
    def _generate_enhanced_parameters_report(self,
                                           model_name: EnhancedCustomModelName,
                                           cas_spec: CASSpecification,
                                           model_path: str) -> str:
        """Generates a detailed report of enhanced model parameters, including CAS metadata.

        Args:
            model_name (EnhancedCustomModelName): The enhanced custom model name object.
            cas_spec (CASSpecification): The CAS specification object.
            model_path (str): The file path to the model.

        Returns:
            str: A multi-line string containing the formatted enhanced parameters report.
        """
        
        lines = [
            "Enhanced Model Parameters Report",
            "=" * 50,
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"System Model ID: {model_name.system_model_id}",
            "",
            "Basic Model Information:",
            f"  Name: {model_name.name}",
            f"  Version: {model_name.version}",
            f"  Type: {model_name.model_type.value}",
            f"  Path: {model_path}",
            f"  Architecture: {cas_spec.base_model.architecture}",
            f"  Parameter Count: {cas_spec.base_model.parameter_count}",
            f"  Context Length: {cas_spec.base_model.context_length}",
            "",
            "CAS Configuration:",
            f"  CAS Version: {cas_spec.metadata.cas_version}",
            f"  Cognitive Profile: {model_name.cognitive_profile}",
            f"  Safety Mode: {model_name.safety_mode}",
            f"  Reasoning Framework: {model_name.reasoning_framework}",
            "",
            "Constitutional Framework:",
            f"  Governor Mode: {cas_spec.constitutional_framework.governor_mode}",
            f"  Enforcement Level: {cas_spec.constitutional_framework.enforcement_level}",
            "",
            "Memory Configuration:",
            f"  Max Memory: {cas_spec.memory_profile.max_memory_gb}GB",
            f"  Management: {cas_spec.memory_profile.memory_management}",
            "",
            "Runtime Adaptation:",
            f"  Enabled: {cas_spec.runtime_adaptation.enabled}",
            f"  Frequency: {cas_spec.runtime_adaptation.adaptation_frequency}",
            f"  Learning Rate: {cas_spec.runtime_adaptation.learning_rate_global}",
            "",
            "Constitutional Principles:"
        ]
        
        # Add constitutional principles
        for category, principles in cas_spec.constitutional_framework.safety_principles.items():
            lines.append(f"  {category.title()}:")
            for principle in principles:
                lines.append(f"    - {principle}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _cas_spec_to_dict(self, cas_spec: CASSpecification) -> Dict[str, Any]:
        """Converts a CASSpecification object into a dictionary representation.

        This method extracts key metadata and base model information from the CAS specification
        to facilitate reporting or serialization.

        Args:
            cas_spec (CASSpecification): The CAS specification object to convert.

        Returns:
            Dict[str, Any]: A dictionary containing the metadata and base model details
                            of the CAS specification.
        """
        return {
            'metadata': {
                'system_model_id': cas_spec.metadata.system_model_id,
                'custom_name': cas_spec.metadata.custom_name,
                'version': cas_spec.metadata.version,
                'cas_version': cas_spec.metadata.cas_version
            },
            'base_model': {
                'path': cas_spec.base_model.path,
                'type': cas_spec.base_model.type,
                'architecture': cas_spec.base_model.architecture,
                'parameter_count': cas_spec.base_model.parameter_count
            }
        }
    
    def _constitutional_framework_to_dict(self, cas_spec: CASSpecification) -> Dict[str, Any]:
        """Converts the constitutional framework of a CAS specification to a dictionary.

        This method extracts details related to the governance, enforcement, safety principles,
        and content filters defined within the CAS constitutional framework.

        Args:
            cas_spec (CASSpecification): The CAS specification object containing the framework.

        Returns:
            Dict[str, Any]: A dictionary representing the constitutional framework.
        """
        return {
            'governor_mode': cas_spec.constitutional_framework.governor_mode,
            'enforcement_level': cas_spec.constitutional_framework.enforcement_level,
            'safety_principles': cas_spec.constitutional_framework.safety_principles,
            'content_filters': cas_spec.constitutional_framework.content_filters
        }
    
    def _memory_profile_to_dict(self, cas_spec: CASSpecification) -> Dict[str, Any]:
        """Converts the memory profile of a CAS specification to a dictionary."""
        memory_dict = {
            'max_memory_gb': cas_spec.memory_profile.max_memory_gb,
            'memory_management': cas_spec.memory_profile.memory_management,
        }
        
        # Check if swap_enabled attribute exists before accessing it
        if hasattr(cas_spec.memory_profile, 'swap_enabled'):
            memory_dict['swap_enabled'] = cas_spec.memory_profile.swap_enabled
        else:
            # Default value or alternative attribute
            memory_dict['swap_enabled'] = False
        
        return memory_dict
    
    def _validate_enhanced_configuration(self, 
                                       config: Dict[str, Any],
                                       model_type: EnhancedCustomModelType) -> Tuple[List[str], List[str]]:
        """Validate enhanced configuration against CAS and model requirements."""
        errors = []
        warnings = []

        # 1. Critical Path and Type Validation
        if not config.get('model_path') or not isinstance(config.get('model_path'), str):
            errors.append("Critical: 'model_path' is missing or not a string.")
        
        # Type checking for model_type
        if not isinstance(model_type, EnhancedCustomModelType):
            errors.append(f"Critical: Invalid model_type provided: {model_type}")
            return errors, warnings # Stop validation if type is wrong

        # 2. CAS-Specific Validations
        if model_type.value.startswith('cas_'):
            # Cognitive Profile Validation
            cognitive_profile = config.get('cognitive_profile')
            if not cognitive_profile:
                warnings.append("CAS Warning: 'cognitive_profile' not specified. Applying system default.")
            elif not isinstance(cognitive_profile, str) or not cognitive_profile.strip():
                errors.append("CAS Error: 'cognitive_profile' must be a non-empty string.")

            # Safety Mode Validation
            safety_mode = config.get('safety_mode')
            if not safety_mode:
                warnings.append("CAS Warning: 'safety_mode' not specified. Applying system default.")
            else:
                try:
                    SafetyMode(safety_mode) # Validate against the enum
                except ValueError:
                    errors.append(f"CAS Error: Invalid 'safety_mode': {safety_mode}. Must be one of {list(SafetyMode.__members__)}.")

            # Constitutional Principles Validation
            principles = config.get('constitutional_principles', [])
            if not isinstance(principles, list) or not all(isinstance(p, str) for p in principles):
                errors.append("CAS Error: 'constitutional_principles' must be a list of strings.")

        # 3. General Configuration Hygiene
        if 'name' not in config or not config['name'].strip():
            errors.append("Configuration Error: Model 'name' cannot be empty.")

        if 'version' in config and not isinstance(config['version'], str):
            warnings.append("Configuration Warning: 'version' should be a string, but it is not.")

        # 4. Check for unrecognized or deprecated keys
        allowed_keys = {
            'model_path', 'name', 'version', 'description', 'cognitive_profile', 
            'safety_mode', 'constitutional_principles', 'reasoning_framework', 
            'export_ollama'
        }
        extra_keys = set(config.keys()) - allowed_keys
        if extra_keys:
            warnings.append(f"Configuration Warning: Unrecognized keys found: {list(extra_keys)}. They will be ignored.")

        return errors, warnings
    
    def _save_enhanced_config_report(self, report: EnhancedModelConfigReport) -> Path:
        """Save enhanced configuration report"""
        report_filename = f"enhanced_{report.model_name}_{report.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.base_directory / "reports" / report_filename
        
        # Convert report to serializable format
        report_data = {
            'report_id': report.report_id,
            'model_name': report.model_name,
            'model_type': report.model_type.value,
            'configuration': report.configuration,
            'created_at': report.created_at.isoformat(),
            'created_by': report.created_by,
            'validation_status': report.validation_status,
            'errors': report.errors,
            'warnings': report.warnings,
            'artifacts': report.artifacts,
            'system_model_id': report.system_model_id,
            'cas_specification': report.cas_specification,
            'constitutional_framework': report.constitutional_framework,
            'cognitive_architecture': report.cognitive_architecture,
            'memory_profile': report.memory_profile,
            'ollama_export_path': report.ollama_export_path
        }
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"Saved enhanced configuration report: {report_path}")
            return report_path
        except IOError as e:
            logger.error(f"Error saving enhanced configuration report to {report_path}: {e}")
            raise # Re-raise the exception after logging


class EnhancedModelFileDirectory(ModelFileDirectory):
    """Manages the directory structure for enhanced model files, including CAS-specific paths.

    This class extends the base ModelFileDirectory to incorporate additional directories
    required for the CAS (Cognitive Architecture System) specifications, Ollama exports,
    constitutional configurations, and cognitive profiles.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """Initializes the EnhancedModelFileDirectory.

        Args:
            base_path (Union[str, Path]): The base directory path where model files and
                                          CAS-related assets will be stored.
        """
        super().__init__(base_path)
        
        # CAS-specific directories
        self.cas_dir = self.base_path / "cas_specifications"
        self.ollama_exports_dir = self.base_path / "ollama_exports"
        self.constitutional_dir = self.base_path / "constitutional_configs"
        self.cognitive_profiles_dir = self.base_path / "cognitive_profiles"
    
    def setup_directory_structure(self) -> None:
        """Creates the complete enhanced directory structure, including CAS-specific directories.

        This method calls the base class's setup_directory_structure and then creates
        additional directories for CAS specifications, Ollama exports, constitutional
        configurations, and cognitive profiles, ensuring all necessary paths exist.
        """
        super().setup_directory_structure()
        
        # Additional CAS directories
        cas_directories = [
            self.cas_dir,
            self.ollama_exports_dir,
            self.constitutional_dir,
            self.cognitive_profiles_dir
        ]
        
        for directory in cas_directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created CAS directory: {directory}")
    
    def get_cas_file_path(self, model_name: str) -> Path:
        """Generates the file path for a CAS specification file.

        Args:
            model_name (str): The name of the model for which to generate the CAS file path.

        Returns:
            Path: The absolute path to the CAS specification file (e.g., `cas_specifications/model_name.cas.yml`).
        """
        return self.cas_dir / f"{model_name}.cas.yml"
    
    def get_ollama_export_path(self, model_name: str) -> Path:
        """Generates the file path for an Ollama export file.

        Args:
            model_name (str): The name of the model for which to generate the Ollama export file path.

        Returns:
            Path: The absolute path to the Ollama export file (e.g., `ollama_exports/model_name.modelfile`).
        """
        return self.ollama_exports_dir / f"{model_name}.modelfile"


# Enhanced convenience functions that work with both base and CAS features
async def create_enhanced_ollama_model(name: str,
                                     base_model: str = "llama3.1:8b-instruct",
                                     cognitive_profile: str = "analytical",
                                     safety_mode: str = "balanced",
                                     use_cas: bool = True,
                                     **kwargs) -> Union[Tuple[Dict[str, Path], EnhancedModelConfigReport], CustomModelFile]:
    """Creates an enhanced Ollama model with optional CAS features.

    This asynchronous function generates a model file for Ollama, optionally integrating
    CAS (Cognitive Architecture System) features based on the `use_cas` flag.
    It leverages the EnhancedGenerateModelFile class for CAS-enabled model creation
    or falls back to the base model_creation for standard Ollama models.

    Args:
        name (str): The name of the model to create.
        base_model (str): The base Ollama model to use (e.g., "llama3.1:8b-instruct").
        cognitive_profile (str): The cognitive profile to apply if CAS is enabled.
                                 Defaults to "analytical".
        safety_mode (str): The safety mode to apply if CAS is enabled.
                           Defaults to "balanced".
        use_cas (bool): A flag indicating whether to enable CAS features for the model.
                        Defaults to True.
        **kwargs: Additional keyword arguments to pass to the model generation process,
                  such as constitutional principles or other CAS-specific configurations.

    Returns:
        Union[Tuple[Dict[str, Path], EnhancedModelConfigReport], CustomModelFile]:
            If `use_cas` is True, returns a tuple containing:
                - A dictionary mapping file types (e.g., 'cas_specification', 'traditional_modelfile')
                  to their generated file paths.
                - An EnhancedModelConfigReport object detailing the model's configuration and validation status.
            If `use_cas` is False, returns a CustomModelFile object representing the base Ollama model.
    """
    if use_cas:
        generator = EnhancedGenerateModelFile()
        return await generator.generate_cas_enhanced_model(
            name=name,
            model_path=base_model,
            model_type="ollama",
            cognitive_profile=cognitive_profile,
            safety_mode=safety_mode,
            export_ollama=True,
            **kwargs
        )
    else:
        # Fallback to base model creation
        from model_creation import create_ollama_model
        return await create_ollama_model(name=name, base_model=base_model, **kwargs)


async def create_enhanced_gguf_model(name: str,
                                   model_path: str,
                                   cognitive_profile: str = "analytical",
                                   safety_mode: str = "balanced",
                                   use_cas: bool = True,
                                   **kwargs) -> Union[Tuple[Dict[str, Path], EnhancedModelConfigReport], CustomModelFile]:
    """Creates an enhanced GGUF model with optional CAS features.

    This asynchronous function generates a model file for GGUF, optionally integrating
    CAS (Cognitive Architecture System) features based on the `use_cas` flag.
    It leverages the EnhancedGenerateModelFile class for CAS-enabled model creation
    or falls back to the base model_creation for standard GGUF models.

    Args:
        name (str): The name of the model to create.
        model_path (str): The file path to the GGUF model.
        cognitive_profile (str): The cognitive profile to apply if CAS is enabled.
                                 Defaults to "analytical".
        safety_mode (str): The safety mode to apply if CAS is enabled.
                           Defaults to "balanced".
        use_cas (bool): A flag indicating whether to enable CAS features for the model.
                        Defaults to True.
        **kwargs: Additional keyword arguments to pass to the model generation process,
                  such as constitutional principles or other CAS-specific configurations.

    Returns:
        Union[Tuple[Dict[str, Path], EnhancedModelConfigReport], CustomModelFile]:
            If `use_cas` is True, returns a tuple containing:
                - A dictionary mapping file types (e.g., 'cas_specification', 'traditional_modelfile')
                  to their generated file paths.
                - An EnhancedModelConfigReport object detailing the model's configuration and validation status.
            If `use_cas` is False, returns a CustomModelFile object representing the base GGUF model.
    """
    if use_cas:
        generator = EnhancedGenerateModelFile()
        return await generator.generate_cas_enhanced_model(
            name=name,
            model_path=model_path,
            model_type="gguf",
            cognitive_profile=cognitive_profile,
            safety_mode=safety_mode,
            export_ollama=True,
            **kwargs
        )
    else:
        # Create base model with GGUF type
        generator = BaseGenerateModelFile()
        generator.setup_model_file_directory()
        return await generator.generate_complete_model_file(
            name=name,
            model_type=BaseCustomModelType.GGUF,
            model_path=model_path,
            **kwargs
        )


# Utility functions for interoperability
def convert_base_to_enhanced_model_name(base_model: BaseCustomModelName, 
                                       cognitive_profile: str = "analytical",
                                       safety_mode: str = "balanced") -> EnhancedCustomModelName:
    """Converts a BaseCustomModelName object to an EnhancedCustomModelName object.

    This utility function facilitates interoperability between the base model naming
    convention and the enhanced CAS-enabled naming convention by adding CAS-specific
    metadata such as cognitive profile and safety mode.

    Args:
        base_model (BaseCustomModelName): The base model name object to convert.
        cognitive_profile (str): The cognitive profile to assign to the enhanced model.
                                 Defaults to "analytical".
        safety_mode (str): The safety mode to assign to the enhanced model.
                           Defaults to "balanced".

    Returns:
        EnhancedCustomModelName: The converted enhanced model name object.
    """
    return EnhancedCustomModelName(
        name=base_model.name,
        model_type=EnhancedCustomModelType.from_base_type(base_model.model_type),
        version=base_model.version,
        description=base_model.description,
        created_at=base_model.created_at,
        system_model_id=base_model.system_model_id,
        cognitive_profile=cognitive_profile,
        safety_mode=safety_mode
    )


def convert_enhanced_to_base_model_name(enhanced_model: EnhancedCustomModelName) -> BaseCustomModelName:
    """Converts an EnhancedCustomModelName object back to a BaseCustomModelName object.

    This utility function facilitates compatibility by stripping away CAS-specific metadata,
    allowing the enhanced model name to be used in contexts that only understand the base
    model naming convention.

    Args:
        enhanced_model (EnhancedCustomModelName): The enhanced model name object to convert.

    Returns:
        BaseCustomModelName: The converted base model name object.
    """
    return enhanced_model.to_base_model_name()

# Example usage and integration test
if __name__ == "__main__":
    import asyncio
    
    async def test_enhanced_integration():
        """Test the enhanced integration system"""
        try:
            # Create enhanced GGUF model
            result = await create_enhanced_gguf_model(
                name="Enhanced Revolutionary Model",
                model_path="./models/deepseek-r1-7b.gguf",
                cognitive_profile="analytical",
                safety_mode="balanced",
                description="Test of enhanced CAS integration",
                constitutional_principles=[
                    "Always prioritize user safety and wellbeing",
                    "Provide accurate and helpful information",
                    "Respect intellectual property and privacy"
                ]
            )
            
            # Handle both return types
            if isinstance(result, tuple):
                paths, report = result
                logger.info("Enhanced Integration Test Results:")
                logger.info(f"  Model Name: {report.model_name}")
                logger.info(f"  System Model ID: {report.system_model_id}")
                logger.info(f"  Validation Status: {report.validation_status}")
                logger.info(f"  CAS Enabled: {report.has_cas_data()}")
                
                logger.info("\nGenerated Files:")
                for file_type, path in paths.items():
                    logger.info(f"    {file_type}: {path}")
                
                logger.info(f"\nArtifact Hashes:")
                for artifact, hash_value in report.artifacts.items():
                    logger.info(f"    {artifact}: {hash_value[:16]}...")
                
                if report.errors:
                    logger.error(f"\nErrors: {report.errors}")
                    for error in report.errors:
                        logger.error(f"    - {error}")
                
                if report.warnings:
                    logger.warning(f"\nWarnings: {report.warnings}")
                    for warning in report.warnings:
                        logger.warning(f"    - {warning}")
            else:
                # Handle base CustomModelFile case
                logger.info("Base Model Creation Test Results:")
                logger.info(f"  Model Name: {result.name}")
                logger.info(f"  File Type: {result.file_type}")
                logger.info(f"  Content Length: {len(result.content)}")
            
            logger.info("\nEnhanced integration test completed successfully!")
            
        except Exception as e:
            logger.error(f"Enhanced integration test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the test
    asyncio.run(test_enhanced_integration())

