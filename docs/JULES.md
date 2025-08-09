# GPT-Ø Agent Task Briefing (JULES.md)
**Document Version:** 2.0
**Last Updated:** 2025-08-08
**Status:** Ground Truth - This document supersedes all previous analysis files.

## 1. Project Overview

**Mission:** Achieve full system integration and validation for the GPT-Ø model.

**Core Principle:** GPT-Ø is a 100% self-contained, self-modifying multimodal AI. All necessary components (model logic, weight computation, tokenizers, output heads, and runtime) are present in the codebase. **No external dependencies, models, or pre-trained weights are required.**

**Your Role (Jules):** As the lead integration engineer, your task is to connect the existing, production-ready components into a single, functional system. You are not required to build major components from scratch. Your focus is on integration, testing, and final validation.

---

## 2. Current System Status

*This status is based on direct codebase analysis and is the definitive source of truth.*

### ✅ **Implemented & Production-Ready Components:**
- **`run.py`:** A comprehensive, production-grade system launcher with a full TUI.
- **`recursive_weights_core.py`:** The complete, functional recursive weight computation system.
- **`bayesian_config_orchestrator.py`:** The complete, functional Bayesian parameter evolution system.
- **`tokenizer_adapter.py` & `tokenizer_mux.py`:** A complete, multimodal tokenizer system supporting 13+ modalities.
- **`tool_output_head.py`:** A complete Universal Tool Control output head.
- **`eyes_outputs.py`:** A complete ISR (Intelligence, Surveillance, Reconnaissance) output head.
- **`ears_outputs.py`:** A complete Spatial Domain Processing output head.
- **`cas/*` Subsystem:** All core components of the Cognitive Architecture Specification (CAS) runtime system are implemented.
- **`gpt_model.py`:** The core transformer architecture is implemented and serves as the central integration point.

### ⏳ **Primary Remaining Tasks (Your Mission):**
- **Integration of Output Heads:** The `eyes` and `ears` output heads are fully implemented but must be integrated into the `gpt_model.py` forward pass and the `tokenizer_adapter.py` modality system.
- **Integration of CAS Runtime:** The `NeuralMemoryRuntime` and constitutional safety features from the `/cas` directory need to be fully wired into the `gpt_model.py` generation loop.
- **Comprehensive Testing:** High-level integration and validation tests are required to ensure all components work together seamlessly and meet the project's strict quality standards.

---

## 3. Your Task Focus & Plan

Your current plan has been approved. The high-level objectives are:

1.  **Update Project Documentation:**
    - [x] Update `docs/TODO.md` to reflect the current integration-focused tasks.
    - [x] Update this `JULES.md` file to serve as the single source of truth.

2.  **Integrate `eyes` and `ears` Output Heads:**
    -   Modify `gpt_model.py` to incorporate the `ISRMasterCoordinator` and `SpatialIntelligenceCoordinator`.
    -   Update the tokenizer system to recognize and process the `EYES` and `EARS` modalities.

3.  **Finalize CAS Integration:**
    -   Complete the integration of `NeuralMemoryRuntime` into the model's generation process.
    -   Ensure the constitutional AI safety constraints from `cas_system.py` are active.

4.  **Develop Integration & Validation Tests:**
    -   Create new test files (`test/test_integration.py`, `test/test_cas_integration.py`).
    -   Write end-to-end tests for all major modalities and subsystems.
    -   Run the full test suite and ensure all tests pass with ≥90% coverage.

5.  **Final System Validation & Submission:**
    -   Use `run.py` to manually validate the fully integrated system.
    -   Run all validation scripts.
    -   Submit the final, tested codebase.

---

## 4. Development Guidelines

**CRITICAL:** You must adhere to the strict, non-negotiable coding guidelines detailed below. The goal is production-ready, secure, high-performance, and maintainable code. **No partial implementations.**

### Code Quality Standards (Summary)

- **Definition of Done (DoD):** A change is complete only if it is fully functional, has ≥90% test coverage (lines and branches), passes all static analysis (lint, type-check), and meets all security and performance requirements.
- **Testing:** Unit, integration, and end-to-end tests are mandatory. The test flake budget is zero.
- **Security:** No hardcoded secrets. All inputs must be validated and outputs encoded. SAST and SCA scans must pass with zero critical/high findings.
- **Performance:** Hot paths must be benchmarked. No performance regressions >2% without approval.
- **Documentation:** All public APIs, classes, and functions must have comprehensive docstrings.
- **Code Review:** All changes require at least one approving review from a code owner. No self-approval.

**(For the complete, detailed list of all 12 mandatory practices and guidelines, refer to the original `gpt_null_integration_analysis.md` document, as these standards remain in effect.)**
