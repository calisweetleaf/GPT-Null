"""
Somnus Sovereign Kernel - Model Creation and Customization Module
==================================================================

Focused on custom model file generation, template management, and configuration auditing.
Provides unique functionality for creating model files (like Ollama MODELFILEs) and managing
custom model configurations without overlapping with model_loader.py or model_schemas.py.

Core Features:
- Custom model file generation and templating
- Custom model name, modelfile, and other model details
- Custom Model Report generation that includes metadata, custom system_model_id, base model name, user chosen name, a brief layout of the MODELFILE turned into a "Bio" style breakdown of the model, and then an SHA-256 hash's of all created files such as the MODELFILE, model report, model parameters file (base model explanation pre-user customization), and the final model file. Every Virtual Machine will keep track of the model by its custom system_model_id and SHA-256 hash, ensuring that the model is traceable, and the user keeps track of all configs.
- Model directory structure management
- Model File Template management
- Model file generation with custom templates.
- Model Name Customization and Management # Future development will tie this in with the actual LLM like the autonomous_prompt_evolution.py file, eventually allowing the model to create its own name, model files, and configurations.
- Model configuration report (model_parameters.txt) generation for base specs of any model. This is then used, along with the custom MODELFILE, system_model_id, and SHA-256 hash to create a unique model report that is then used to track the model in the system, along with its configurations. This has no effect on functionality this is purely for a attempt to "un-blackbox" the model, and allow for a user to see what the model is, and how it works.
- Model file finalization and validation
- Configuration reporting and auditing
- Template-based model metadata handling
- Bridge between different model providers (MSTY, LMStudio, Ollama)
- ModelParameterReport Class for generating model parameters file that include original model name, parameter count, context length, output tokens, modalities, capabailities etc.
- Algorithm for SHA-256 hashing of a model once its loaded by model_loader.py
"""

import os
import logging
import uuid
import yaml
import json
import hashlib
import asyncio
import weakref
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# Mock interfaces for non-existent dependencies (production-ready fallbacks)
from typing import Protocol

class LLMProvider(Protocol):
    """Protocol for LLM provider interface with comprehensive method definitions."""
    
    def generate(self, prompt: str, max_tokens: int = 1024, **kwargs) -> str:
        """Generate text response from prompt."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get provider configuration."""
        ...
    
    def validate_input(self, prompt: str) -> bool:
        """Validate input prompt."""
        ...

class GhostSecurityConfig:
    """Mock security configuration for compatibility."""
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def dict(self) -> Dict[str, Any]:
        return self.config

class AntiFingerprintingConfig:
    """Mock anti-fingerprinting configuration."""
    def __init__(self, **kwargs):
        self.config = kwargs

class ModelSteganographyConfig:
    """Mock steganography configuration."""
    def __init__(self, **kwargs):
        self.config = kwargs

class AntiDetectionMode(Enum):
    """Anti-detection mode enumeration."""
    DISABLED = "disabled"
    BASIC = "basic" 
    ADVANCED = "advanced"

class FingerprintingVector(Enum):
    """Fingerprinting vector enumeration."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# System constants for model creation
SYSTEM_USER_ID = "Somnus_Operator"
DEFAULT_MODEL_DIR = Path("model_parameters")
DEFAULT_MODELFILE_DIR = DEFAULT_MODEL_DIR / "model_parameters/MODELFILE/"
DEFAULT_REPORTS_DIR = DEFAULT_MODEL_DIR / "model_parameters/reports/"


class CustomModelType(Enum): #replace with Model Path that is generated and used by this file and the model_loader.py
    """Custom model types for local model creation"""
    OLLAMA = "OLLAMA_API_ENDPOINT"
    MSTY = "MSTY_API_ENDPOINT"  # Placeholder for MSTY API endpoint
    LMSTUDIO = "LM_STUDIO_ENDPOINT"  # Placeholder for LMStudio endpoint
    OPEN_ROUTER = "OPEN_ROUTER_API_KEY"  # Placeholder for OpenRouter API endpoint
    OPEN_ROUTER_MODELNAME = "open_router_model_name"  # Placeholder for OpenRouter model name
    LOCAL_PYTHON = "/models/local_python"
    HUGGINGFACE = "HUGGINGFACE_API_ENDPOINT"  # Placeholder for HuggingFace API endpoint
    GGUF = "gguf"
    GGML = "ggml"
    PyTorch = "pytorch"
    TensorFlow = "tensorflow"
    JAX = "jax"
    ONNX = "onnx"
    CUSTOM_NEURAL_NETWORKS = "custom_neural_model"  # General type for custom neural networks
    ALGO = "algorithmic_model"  # For algorithmic models that don't fit neural network paradigms
    DIFFUSION = "diffusion_models" # for native image, audio, video, and other diffusion models.
    GENERATIVE = "generative_models" # for generative models that create new content.
    EMBEDDING = "embedding"  # For models focused on generating embeddings
    CUSTOM_UNIVERSAL_MODEL = "custom_universal_model"  # For custom models that may be user made and consist of anything. If it can be classified as a model, it can be used by the model_creation system and model_loader.py, to be injected into the virtual machine.


# ---------------------------------
# Internal helpers (modular, pure)
# ---------------------------------

def _safe_enum_from_value(value: Any, default: CustomModelType = CustomModelType.CUSTOM_UNIVERSAL_MODEL) -> CustomModelType:
    """Safely convert a string/enum-like value to CustomModelType without raising.
    Falls back to the provided default if the value is unrecognized.
    """
    if isinstance(value, CustomModelType):
        return value
    if isinstance(value, str):
        # try name first
        try:
            return CustomModelType[value]
        except Exception:
            # try value matching
            for e in CustomModelType:
                if e.value == value:
                    return e
    return default


def compute_sha256_bytes(content: bytes) -> str:
    """Compute SHA-256 for raw bytes and return hex digest."""
    h = hashlib.sha256()
    h.update(content)
    return h.hexdigest()


def compute_sha256_text(text: str, encoding: str = "utf-8") -> str:
    """Compute SHA-256 for text input."""
    return compute_sha256_bytes(text.encode(encoding))


def compute_sha256_file(path: Union[str, Path]) -> str:
    """Compute SHA-256 for a file path. Returns empty string if file doesn't exist."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_system_model_id() -> str:
    """Generate a unique, opaque system_model_id."""
    return f"somnus-{uuid.uuid4()}"


@dataclass
class CustomModelName:
    """Custom model name with metadata."""
    name: str
    model_type: CustomModelType
    version: str = "1.0.0"
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    system_model_id: str = field(default_factory=generate_system_model_id)

    def to_filename(self) -> str:
        """Convert to safe filename."""
        safe_name = "".join(c for c in self.name if c.isalnum() or c in ('-', '_')).lower()
        return f"{safe_name}-{self.version}"


@dataclass 
class CustomModelFile:
    """Represents a custom model file with template support."""
    name: str
    content: str
    file_type: str  # "modelfile", "config", "parameters"
    template_variables: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def fill_template(self, **kwargs) -> 'CustomModelFile':
        """Fill template variables in content.
        Replaces placeholders of the form {key}.
        """
        filled_content = self.content
        variables = {**self.template_variables, **kwargs}
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            filled_content = filled_content.replace(placeholder, str(value))
        return CustomModelFile(
            name=self.name,
            content=filled_content,
            file_type=self.file_type,
            template_variables={},
            metadata={**self.metadata, "template_filled": True}
        )

    def apply_final_tweaks(self) -> 'CustomModelFile':
        """Apply final modifications to the model file: header stamp and metadata flag."""
        tweaked_content = self.content
        if not tweaked_content.startswith("#"):
            tweaked_content = f"# Generated by Somnus Model Creation System\n# {datetime.now().isoformat()}\n\n{tweaked_content}"
        return CustomModelFile(
            name=self.name,
            content=tweaked_content,
            file_type=self.file_type,
            template_variables=self.template_variables,
            metadata={**self.metadata, "final_tweaks_applied": True}
        )


@dataclass
class ModelFileTemplate:
    """Template for model file generation."""
    template_id: str
    name: str
    content_template: str
    required_variables: List[str]
    optional_variables: Dict[str, str] = field(default_factory=dict)
    target_type: CustomModelType = CustomModelType.CUSTOM_UNIVERSAL_MODEL
    description: str = ""


@dataclass
class ModelConfigReport:
    """Audit report for model configuration."""
    report_id: str
    model_name: str
    model_type: CustomModelType
    configuration: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = SYSTEM_USER_ID
    validation_status: str = "pending"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # mapping filename -> sha256
    system_model_id: Optional[str] = None
    ghost_security_config: Optional[GhostSecurityConfig] = None


# ============================================================================
# FILE AND DIRECTORY MANAGEMENT
# ============================================================================

class ModelFileDirectory:
    """Manages directories containing model files."""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.modelfile_dir = self.base_path / "MODELFILE"
        self.templates_dir = self.base_path / "templates"
        self.reports_dir = self.base_path / "reports"
        self.configs_dir = self.base_path / "configs"
        self.backups_dir = self.base_path / "backups"
    
    def setup_directory_structure(self) -> None:
        """Create the complete directory structure."""
        directories = [
            self.base_path,
            self.modelfile_dir,
            self.templates_dir,
            self.reports_dir,
            self.configs_dir,
            self.backups_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def list_model_files(self, file_type: Optional[str] = None) -> List[Path]:
        """List all model files in the directory tree, optionally filtered by extension."""
        files: List[Path] = []
        for path in self.modelfile_dir.rglob("*"):
            if path.is_file():
                if file_type is None or path.suffix == f".{file_type}":
                    files.append(path)
        return files
    
    def get_model_file_path(self, model_name: str, file_type: str = "modelfile") -> Path:
        """Get the path for a model file."""
        safe_type = (file_type or "modelfile").lstrip('.')
        return self.modelfile_dir / f"{model_name}.{safe_type}"


class ModelFileLoader:
    """Specialized loader for model files and templates."""
    
    @staticmethod
    async def load_template(template_path: Union[str, Path], timeout: float = 30.0) -> ModelFileTemplate:
        """Load a model file template, with optional YAML frontmatter."""
        import asyncio
        
        path = Path(template_path)
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        # Input validation
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        
        try:
            # Use asyncio timeout for file operations
            async def _read_file():
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            content = await asyncio.wait_for(_read_file(), timeout=timeout)
            # Parse template metadata if present (YAML frontmatter)
            if content.startswith('---\n'):
                parts = content.split('---\n', 2)
                if len(parts) >= 3:
                    metadata = yaml.safe_load(parts[1]) or {}
                    template_content = parts[2]
                else:
                    metadata = {}
                    template_content = content
            else:
                metadata = {}
                template_content = content
            target = _safe_enum_from_value(metadata.get('target_type', 'custom'))
            return ModelFileTemplate(
                template_id=(metadata.get('id') or path.stem),
                name=metadata.get('name', path.stem),
                content_template=template_content,
                required_variables=list(metadata.get('required_variables', [])),
                optional_variables=dict(metadata.get('optional_variables', {})),
                target_type=target,
                description=metadata.get('description', ''),
            )
        except asyncio.TimeoutError:
            logger.error(f"Template loading timed out after {timeout}s: {template_path}")
            raise TimeoutError(f"Template loading timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Failed to load template {template_path}: {e}")
            raise
    
    @staticmethod
    async def load_custom_file(file_path: Union[str, Path]) -> CustomModelFile:
        """Load a custom model file from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return CustomModelFile(
                name=path.stem,
                content=content,
                file_type=path.suffix.lstrip('.') or 'modelfile',
                metadata={'source_path': str(path)}
            )
        except Exception as e:
            logger.error(f"Failed to load model file {file_path}: {e}")
            raise
    
    @staticmethod
    async def save_custom_file(model_file: CustomModelFile, output_path: Union[str, Path]) -> Path:
        """Save a custom model file to disk and return the path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(model_file.content)
            logger.info(f"Saved model file: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save model file {output_path}: {e}")
            raise


# ============================================================================
# MODEL GENERATION AND MANAGEMENT
# ============================================================================

class GenerateModelFile:
    """Main class for generating model files for different providers."""
    
    def __init__(self, base_directory: Optional[Union[str, Path]] = None):
        self.base_directory = Path(base_directory or DEFAULT_MODEL_DIR)
        self.directory_manager = ModelFileDirectory(self.base_directory)
        self.templates_cache: Dict[str, ModelFileTemplate] = {}
    
    def setup_model_file_directory(self) -> None:
        """Set up the complete model file directory structure."""
        self.directory_manager.setup_directory_structure()
        logger.info(f"Model file directory structure created at {self.base_directory}")
    
    def create_model_name(self, name: str, model_type: CustomModelType, version: str = "1.0.0", 
                         description: str = "") -> CustomModelName:
        """Create a custom model name object."""
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        model_type = _safe_enum_from_value(model_type)
        return CustomModelName(
            name=name,
            model_type=model_type,
            version=version,
            description=description
        )
    
    async def load_model_file_template(self, template_name: str) -> ModelFileTemplate:
        """Load model file template with caching."""
        if template_name in self.templates_cache:
            return self.templates_cache[template_name]
        template_path = self.directory_manager.templates_dir / f"{template_name}.template"
        if not template_path.exists():
            await self._create_default_template(template_name, template_path)
        template = await ModelFileLoader.load_template(template_path)
        self.templates_cache[template_name] = template
        return template
    
    async def auto_fill_template(self, template: ModelFileTemplate, model_name: CustomModelName, 
                                **additional_vars) -> CustomModelFile:
        """Automatically fill model file template with provided information."""
        variables = {
            'model_name': model_name.name,
            'model_type': model_name.model_type.value,
            'version': model_name.version,
            'description': model_name.description,
            'created_at': model_name.created_at.isoformat(),
            'system_id': model_name.system_model_id,
            **additional_vars
        }
        missing_vars = [var for var in template.required_variables if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required template variables: {missing_vars}")
        for var, default in template.optional_variables.items():
            if var not in variables:
                variables[var] = default
        custom_file = CustomModelFile(
            name=model_name.to_filename(),
            content=template.content_template,
            file_type="modelfile",
            template_variables=variables,
            metadata={
                'template_id': template.template_id,
                'model_type': model_name.model_type.value,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
        )
        return custom_file.fill_template(**variables)
    
    def final_model_file_tweaks(self, model_file: CustomModelFile) -> CustomModelFile:
        """Apply final tweaks and validation to the model file."""
        tweaked_file = model_file.apply_final_tweaks()
        content = tweaked_file.content
        model_type = tweaked_file.metadata.get('model_type', 'custom')
        if model_type == 'ollama':
            if not content.startswith('FROM '):
                content = f"FROM llama2\n\n{content}"
        elif model_type == 'lmstudio':
            if '"model_type"' not in content:
                content = f'{"{"}"model_type": "causal_lm"{"}"}\n\n{content}'
        return CustomModelFile(
            name=tweaked_file.name,
            content=content,
            file_type=tweaked_file.file_type,
            template_variables=tweaked_file.template_variables,
            metadata={**tweaked_file.metadata, "custom_tweaks_applied": True}
        )
    
    async def generate_complete_model_file(self, name: str, model_type: CustomModelType, 
                                         template_name: Optional[str] = None, 
                                         ghost_security_config: Optional[GhostSecurityConfig] = None, 
                                         **kwargs) -> CustomModelFile:
        """Generate a complete model file from start to finish."""
        model_name = self.create_model_name(name, model_type, **{k: v for k, v in kwargs.items() if k in {"version", "description"}})
        if template_name is None:
            template_name = _safe_enum_from_value(model_type).value
        template = await self.load_model_file_template(template_name)
        filled_file = await self.auto_fill_template(template, model_name, **kwargs)
        final_file = self.final_model_file_tweaks(filled_file)
        output_path = self.directory_manager.get_model_file_path(model_name.to_filename())
        saved_path = await ModelFileLoader.save_custom_file(final_file, output_path)
        # compute hash and attach to metadata
        final_file.metadata['sha256'] = compute_sha256_file(saved_path)
        final_file.metadata['system_model_id'] = model_name.system_model_id
        if ghost_security_config:
            final_file.metadata['ghost_security_config'] = ghost_security_config.dict() # Store as dict for serialization
        logger.info(f"Generated complete model file: {output_path}")
        return final_file
    
    async def _create_default_template(self, template_name: str, template_path: Path) -> None:
        """Create a default template if none exists."""
        default_templates = {
            'ollama': '''---
id: ollama_default
name: Ollama Model Template
required_variables: [model_name, base_model]
optional_variables:
  temperature: "0.7"
  system_prompt: "You are a helpful assistant."
target_type: ollama
description: Default Ollama model file template
---
FROM {base_model}

PARAMETER temperature {temperature}
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM "{system_prompt}"

# Model: {model_name}
# Generated: {created_at}
# Type: {model_type}
''',
            'lmstudio': '''---
id: lmstudio_default  
name: LMStudio Model Template
required_variables: [model_name, model_path]
optional_variables:
  max_tokens: "2048"
  temperature: "0.7"
target_type: lmstudio
description: Default LMStudio model configuration template
---
{
  "model_name": "{model_name}",
  "model_path": "{model_path}",
  "model_type": "causal_lm",
  "max_tokens": {max_tokens},
  "temperature": {temperature},
  "generated_at": "{created_at}",
  "version": "{version}"
}
''',
            'msty': '''---
id: msty_default
name: MSTY Model Template  
required_variables: [model_name, endpoint_url]
optional_variables:
  context_length: "4096"
  temperature: "0.7"
target_type: msty
description: Default MSTY model configuration template
---
model_name: {model_name}
endpoint_url: {endpoint_url}
context_length: {context_length}
temperature: {temperature}
model_type: {model_type}
version: {version}
created_at: {created_at}
'''
        }
        template_content = default_templates.get(template_name, default_templates['ollama'])
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        logger.info(f"Created default template: {template_path}")


# ============================================================================
# PARAMETERS FILE GENERATION (ModelParameterReport)
# ============================================================================

@dataclass
class ModelParameterReport:
    """Structured parameter report for a base model and its characteristics."""
    model_name: str
    original_model_id: Optional[str] = None
    parameter_count: Optional[int] = None
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    modalities: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    provider: Optional[str] = None
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_text(self) -> str:
        lines = [
            f"Model Parameters Report",
            f"Generated: {self.created_at.isoformat()}",
            f"Model Name: {self.model_name}",
        ]
        if self.original_model_id: lines.append(f"Original Model ID: {self.original_model_id}")
        if self.parameter_count is not None: lines.append(f"Parameters: {self.parameter_count}")
        if self.context_length is not None: lines.append(f"Context Length: {self.context_length}")
        if self.max_output_tokens is not None: lines.append(f"Max Output Tokens: {self.max_output_tokens}")
        if self.modalities: lines.append(f"Modalities: {', '.join(self.modalities)}")
        if self.capabilities: lines.append(f"Capabilities: {', '.join(self.capabilities)}")
        if self.provider: lines.append(f"Provider: {self.provider}")
        if self.notes:
            lines.append("")
            lines.append(self.notes)
        return "\n".join(lines)


class GenerateModelParamtersFile:
    """Generate model parameters file (text) capturing baseline specs and metadata."""

    def __init__(self, base_directory: Optional[Union[str, Path]] = None):
        self.base_directory = Path(base_directory or DEFAULT_MODEL_DIR)
        self.directory_manager = ModelFileDirectory(self.base_directory)
        self.templates_cache: Dict[str, ModelFileTemplate] = {}

    async def load_model_file_template(self, template_name: str) -> ModelFileTemplate:
        """Load parameters template using the same mechanism as model templates."""
        if template_name in self.templates_cache:
            return self.templates_cache[template_name]
        template_path = self.directory_manager.templates_dir / f"{template_name}.template"
        if not template_path.exists():
            await self._create_default_parameters_template(template_name, template_path)
        template = await ModelFileLoader.load_template(template_path)
        self.templates_cache[template_name] = template
        return template

    async def auto_fill_template(self, template: ModelFileTemplate, model_name: CustomModelName, 
                                **additional_vars) -> CustomModelFile:
        """Fill a parameters template to produce a parameters text file."""
        variables = {
            'model_name': model_name.name,
            'model_type': model_name.model_type.value,
            'version': model_name.version,
            'created_at': model_name.created_at.isoformat(),
            'system_id': model_name.system_model_id,
            **additional_vars
        }
        missing_vars = [var for var in template.required_variables if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required template variables: {missing_vars}")
        for var, default in template.optional_variables.items():
            if var not in variables:
                variables[var] = default
        model_file = CustomModelFile(
            name=f"{model_name.to_filename()}-parameters",
            content=template.content_template,
            file_type="txt",
            template_variables=variables,
            metadata={'template_id': template.template_id, 'model_type': model_name.model_type.value}
        )
        return model_file.fill_template(**variables)

    def final_model_file_tweaks(self, model_file: CustomModelFile) -> CustomModelFile:
        """Apply simple header stamp to parameters file."""
        return model_file.apply_final_tweaks()

    async def generate_complete_parameters_file(
        self,
        model_name: CustomModelName,
        parameters: ModelParameterReport,
        template_name: str = "parameters_default"
    ) -> Path:
        """Generate and save a full parameters file. Returns saved path."""
        self.directory_manager.setup_directory_structure()
        template = await self.load_model_file_template(template_name)
        parameters_text = parameters.to_text()
        # embed the parameters_text into template via variable
        filled = await self.auto_fill_template(template, model_name, parameters_text=parameters_text)
        final_file = self.final_model_file_tweaks(filled)
        output_path = self.directory_manager.get_model_file_path(final_file.name, file_type="txt")
        saved_path = await ModelFileLoader.save_custom_file(final_file, output_path)
        return saved_path

    async def _create_default_parameters_template(self, template_name: str, template_path: Path) -> None:
        """Create a default parameters template if none exists."""
        content = '''---
id: parameters_default
name: Default Parameters Template
required_variables: [model_name, parameters_text]
optional_variables: {}
target_type: custom
description: Default template for model parameters report
---
# Parameters for {model_name}

{parameters_text}
'''
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Created default parameters template: {template_path}")


# ============================================================================
# REPORTING AND AUDITING
# ============================================================================

class ModelConfigReportManager:
    """Manages model configuration reports and auditing."""
    
    def __init__(self, reports_directory: Union[str, Path]):
        self.reports_dir = Path(reports_directory)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def create_model_config_report(self, model_name: str, model_type: CustomModelType, 
                                 configuration: Dict[str, Any]) -> ModelConfigReport:
        """Create a new model configuration report with validation."""
        report = ModelConfigReport(
            report_id=str(uuid.uuid4()),
            model_name=model_name,
            model_type=_safe_enum_from_value(model_type),
            configuration=configuration
        )
        report.errors, report.warnings = self._validate_configuration(configuration, report.model_type)
        report.validation_status = "passed" if not report.errors else "failed"
        return report
    
    def save_report(self, report: ModelConfigReport) -> Path:
        """Save a model configuration report to JSON and return the path."""
        report_filename = f"{report.model_name}_{report.model_type.value}_{report.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.reports_dir / report_filename
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
        }
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Saved model configuration report: {report_path}")
        return report_path
    
    def attach_artifact_hashes(self, report: ModelConfigReport, artifact_paths: Dict[str, Union[str, Path]]) -> None:
        """Compute and attach SHA-256 hashes for given artifact paths to the report."""
        for name, path in artifact_paths.items():
            report.artifacts[name] = compute_sha256_file(path)

    def set_system_model_id(self, report: ModelConfigReport, system_model_id: str) -> None:
        """Attach system_model_id to the report."""
        report.system_model_id = system_model_id
    
    def _validate_configuration(self, config: Dict[str, Any], model_type: CustomModelType) -> Tuple[List[str], List[str]]:
        """Validate model configuration and return errors and warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not config.get('model_name'):
            errors.append("model_name is required")
        if model_type == CustomModelType.OLLAMA:
            if not config.get('base_model'):
                errors.append("base_model is required for Ollama models")
        elif model_type == CustomModelType.LMSTUDIO:
            if not config.get('model_path'):
                errors.append("model_path is required for LMStudio models")
        elif model_type == CustomModelType.MSTY:
            if not config.get('endpoint_url'):
                errors.append("endpoint_url is required for MSTY models")
        temperature = config.get('temperature', 0.7)
        if isinstance(temperature, (int, float)) and (temperature < 0 or temperature > 2):
            warnings.append("temperature should be between 0 and 2")
        return errors, warnings


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_ollama_model(name: str, base_model: str = "llama2", ghost_security_config: Optional[GhostSecurityConfig] = None, **kwargs) -> CustomModelFile:
    """Convenience function to create an Ollama model file."""
    generator = GenerateModelFile()
    generator.setup_model_file_directory()
    return await generator.generate_complete_model_file(
        name=name,
        model_type=CustomModelType.OLLAMA,
        base_model=base_model,
        ghost_security_config=ghost_security_config,
        **kwargs
    )


async def create_lmstudio_model(name: str, model_path: str, ghost_security_config: Optional[GhostSecurityConfig] = None, **kwargs) -> CustomModelFile:
    """Convenience function to create an LMStudio model configuration."""
    generator = GenerateModelFile()
    generator.setup_model_file_directory()
    return await generator.generate_complete_model_file(
        name=name,
        model_type=CustomModelType.LMSTUDIO,
        model_path=model_path,
        ghost_security_config=ghost_security_config,
        **kwargs
    )


async def create_msty_model(name: str, endpoint_url: str, ghost_security_config: Optional[GhostSecurityConfig] = None, **kwargs) -> CustomModelFile:
    """Convenience function to create an MSTY model configuration."""
    generator = GenerateModelFile()
    generator.setup_model_file_directory()
    return await generator.generate_complete_model_file(
        name=name,
        model_type=CustomModelType.MSTY,
        endpoint_url=endpoint_url,
        ghost_security_config=ghost_security_config,
        **kwargs
    )

if __name__ == '__main__':
    import asyncio

    async def main():
        """Main function to demonstrate model creation capabilities."""
        logger.info("Starting model creation demonstration...")

        # Setup directory structure
        generator = GenerateModelFile()
        generator.setup_model_file_directory()

        # --- 1. Create a standard Ollama model ---
        logger.info("\n--- Generating Standard Ollama Model ---")
        try:
            ollama_model_file = await create_ollama_model(
                name="MyOllamaModel",
                base_model="llama3.1:latest",
                system_prompt="You are a master of Python programming."
            )
            logger.info(f"Ollama model file created: {ollama_model_file.name}")
            logger.info(f"  System ID: {ollama_model_file.metadata.get('system_model_id')}")
            logger.info(f"  SHA256: {ollama_model_file.metadata.get('sha256')}")

        except Exception as e:
            logger.error(f"Failed to create Ollama model: {e}")

        # --- 2. Create a standard LMStudio model ---
        logger.info("\n--- Generating Standard LMStudio Model ---")
        try:
            lmstudio_model_file = await create_lmstudio_model(
                name="MyLMStudioModel",
                model_path="/models/gguf/my-lm-studio-model.gguf",
                max_tokens="4096"
            )
            logger.info(f"LMStudio model file created: {lmstudio_model_file.name}")
            logger.info(f"  System ID: {lmstudio_model_file.metadata.get('system_model_id')}")
            logger.info(f"  SHA256: {lmstudio_model_file.metadata.get('sha256')}")

        except Exception as e:
            logger.error(f"Failed to create LMStudio model: {e}")

        # --- 3. Generate a complete model with parameters and reports ---
        logger.info("\n--- Generating Complete Model Package ---")
        try:
            # Create a custom model name
            model_name_obj = generator.create_model_name(
                name="MyCompleteModel",
                model_type=CustomModelType.GGUF,
                version="1.2.3",
                description="A complete model with all artifacts."
            )

            # Generate the main model file (modelfile)
            model_file = await generator.generate_complete_model_file(
                name=model_name_obj.name,
                model_type=model_name_obj.model_type,
                version=model_name_obj.version,
                description=model_name_obj.description,
                system_id=model_name_obj.system_model_id,
                base_model="phi3:latest"
            )
            model_file_path = generator.directory_manager.get_model_file_path(model_name_obj.to_filename())
            await ModelFileLoader.save_custom_file(model_file, model_file_path)

            # Generate a parameters report
            params_generator = GenerateModelParamtersFile()
            parameter_report_obj = ModelParameterReport(
                model_name=model_name_obj.name,
                original_model_id="phi3:latest",
                parameter_count=3000000000,
                context_length=8192,
                modalities=["text"],
                capabilities=["text-generation", "reasoning"],
                provider="Microsoft"
            )
            params_file_path = await params_generator.generate_complete_parameters_file(
                model_name=model_name_obj,
                parameters=parameter_report_obj
            )

            # Create and save a configuration report
            report_manager = ModelConfigReportManager(generator.directory_manager.reports_dir)
            config_report = report_manager.create_model_config_report(
                model_name=model_name_obj.name,
                model_type=model_name_obj.model_type,
                configuration={
                    'base_model': 'phi3:latest',
                    'temperature': 0.8,
                    'custom_prompt': 'You are a helpful AI assistant for science.'
                }
            )
            
            # Attach artifact hashes
            artifacts_to_hash = {
                'modelfile': model_file_path,
                'parameters_report': params_file_path
            }
            report_manager.attach_artifact_hashes(config_report, artifacts_to_hash)
            report_manager.set_system_model_id(config_report, model_name_obj.system_model_id)
            
            report_path = report_manager.save_report(config_report)

            logger.info("Complete model package generated successfully:")
            logger.info(f"  Model File: {model_file_path}")
            logger.info(f"  Parameters Report: {params_file_path}")
            logger.info(f"  Config Report: {report_path}")
            logger.info(f"  System ID: {config_report.system_model_id}")
            logger.info(f"  Artifact Hashes: {config_report.artifacts}")

        except Exception as e:
            logger.error(f"Failed to generate complete model package: {e}")

        logger.info("\nModel creation demonstration finished.")

    asyncio.run(main())