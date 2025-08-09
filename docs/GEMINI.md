# GPT-Ø Project Overview

This document provides a comprehensive overview of the GPT-Ø project, a self-modifying, multimodal AI system designed for advanced reasoning, tool synthesis, and autonomous operation, with a focus on running efficiently on consumer hardware.

## Project Purpose and Vision

GPT-Ø (GPT-Zero) represents a novel approach to AI, moving beyond traditional pre-training paradigms. Its core innovation lies in **interaction-driven evolution**, where the model continuously adapts and develops capabilities through real-time interaction rather than static datasets. The project aims to create a highly capable, agentic AI that can operate effectively within resource-constrained environments, particularly consumer-grade hardware (e.g., 8GB RAM). It is being developed with a strong emphasis on defensive model applications and operator agentic self-sustaining capabilities.

## Core Technologies and Architecture

The GPT-Ø system is built primarily using **PyTorch** and features a highly modular and dynamic architecture. Key components and their roles include:

*   **`gpt_model.py` (GPT-Ø Core):** The central self-modifying transformer model. It integrates all other components and orchestrates the multimodal processing, reasoning, and output generation.
*   **`recursive_weights_core.py` (Recursive Weight System):** Implements a revolutionary dynamic weight computation mechanism. Instead of static weights, GPT-Ø's weights are computed contextually in real-time, enabling true architectural self-modification and efficient memory usage.
*   **`bayesian_config_orchestrator.py` (Bayesian Configuration Orchestrator):** A dynamic hyperparameter optimization and architectural adaptation system. It uses Bayesian inference to explore parameter space, tune hyperparameters based on performance feedback, and facilitate architectural plasticity (e.g., dynamic layer adjustments).
*   **`cas/neural_memory_runtime.py` (Neural Memory Runtime):** A breakthrough memory management system that replaces the traditional KV-cache. It employs neural compression, dynamic sparse attention, and quantum-inspired memory states to achieve massive memory efficiency, allowing large models to run on limited RAM (e.g., 8GB).
*   **`extra_output_heads/tool_output_head.py` (Universal Tool Control Output Head):** Enables GPT-Ø to synthesize novel tools, discover and profile external systems, adapt to various interfaces, and control multi-domain systems (digital, mechanical, analog, electromagnetic, optical).
*   **`extra_output_heads/eyes_outputs.py` (ISR Master Coordinator):** Provides Intelligence, Surveillance, and Reconnaissance (ISR) capabilities with autonomous defensive authority and sovereign operational control.
*   **`extra_output_heads/ears_outputs.py` (Spatial Master Coordinator):** Offers complete spatial domain processing, including depth perception, stereoscopic analysis, thermal imaging, radar, sonar, and IMU integration for tactical coordination.
*   **`cas/cas_system.py` (Constitutional AI System):** Integrates a constitutional safety framework, ensuring that the model's behavior adheres to predefined ethical and safety principles.
*   **`tokenizer_mux.py` and `tokenizer_adapter.py`:** Handle multimodal tokenization, allowing the model to process and generate content across various modalities (text, image, audio, structured data, etc.).
*   **Mixture of Experts (MoE):** Integrated within `gpt_model.py` to route tokens to specialized feed-forward networks, enhancing performance and efficiency.
*   **Chain of Thought (CoT) Processor:** A neural component within `gpt_model.py` that facilitates complex reasoning steps, tracks stability, and manages knowledge.

## Building and Running

The project is primarily Python-based and relies on `pip` for dependency management.

### Dependencies

All required Python packages are listed in `requirements.txt`. To install them, navigate to the project root and run:

```bash
pip install -r requirements.txt
```

### Environment Setup

The project includes scripts for environment setup, particularly for the "JULES" environment (likely a specific deployment or testing setup).

*   `setup_jules_env.sh` (for Linux/macOS)
*   `jules_setup_script.py` (cross-platform Python script)

These scripts may handle virtual environment creation and initial configuration.

### Running the System

The main entry point for launching GPT-Ø is `run.py`. It provides a terminal user interface (TUI) for interaction and monitoring.

To start the system:

```bash
python run.py
```

You can also specify a custom configuration file or enable debug logging:

```bash
python run.py --config path/to/your_config.json
python run.py --debug
```

### Testing

The project uses `pytest` for its test suite. Tests are located in the `test/` directory.

To run all tests:

```bash
pytest
```

Specific test files can be run by providing their path:

```bash
pytest test/test_gpt_zero.py
```

## Development Conventions

*   **Language:** Python 3.9+
*   **Framework:** PyTorch 2.0+
*   **Type Hinting:** Extensive use of Python type hints for code clarity and maintainability.
*   **Code Formatting:** `black` and `isort` are used to maintain consistent code style.
*   **Linting and Static Analysis:** `flake8`, `mypy`, and `ruff` are employed for code quality checks.
*   **Logging:** Comprehensive logging is implemented using Python's `logging` module, with `RichHandler` for enhanced terminal output and correlation IDs for traceability.
*   **Security:** Adherence to OWASP Top 10 principles is a stated goal, particularly for the recursive weights system.
*   **Test Coverage:** A target of ≥90% line and branch test coverage is aimed for, ensuring robust and reliable code.
*   **Documentation:** Key architectural components and functionalities are documented within the code via docstrings and in Markdown files (`docs/`, `MODELCARD.md`, `TODO.md`).