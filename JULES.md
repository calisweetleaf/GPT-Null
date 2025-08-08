# Project Context

## Inspiration: Google's Jules Asynchronous AI Agent Architecture

This project draws architectural inspiration from Google's Jules (JULES) - an autonomous asynchronous AI coding agent powered by Gemini 2.5 Pro. Key architectural patterns adopted:

- **Asynchronous Operation**: Like Jules, GPT-Ø operates independently with human oversight
- **Full Repository Context**: Complete codebase analysis before making changes
- **Multi-Step Planning**: Detailed execution plans with human approval checkpoints
- **Agentic Architecture**: Autonomous decision-making within defined parameters
- **Secure Isolation**: Operations in contained environments (VM for Jules, memory isolation for GPT-Ø)
- **Developer-in-the-Loop**: Human control with transparency at every step

## GPT-Ø: Completely Self-Contained Architecture

### Core Principle: ZERO External Dependencies

**GPT-Ø is 100% self-contained with NO external models, weights, tensors, embeddings, or pre-trained components.** Everything needed for operation exists within the codebase itself. This is not a wrapper around existing models - it IS the complete model.

### Complete Model Components (Self-Contained)

1. **Core Model Logic** (`gpt_model.py`): The complete transformer architecture, attention mechanisms, layer definitions, forward pass, generation logic
2. **Dynamic Weight Computation** (`recursive_weights_core.py`): Mathematical weight generation using recursive formalism {B, Φ, R, T, ε} - NO static weights stored
3. **Multimodal Processing** (`tokenizer_mux.py` + `tokenizer_adapter.py`): All 13+ modality tokenizers, encoders, decoders built-in
4. **Specialized Output Heads** (`extra_output_heads/`):
   - `tool_output_head.py`: Universal tool synthesis and system control
   - `eyes_outputs.py`: ISR (Intelligence, Surveillance, Reconnaissance) processing
   - `ears_outputs.py`: Spatial domain processing and tactical coordination

### CAS Runtime System: The Engine That Powers It All

The `/cas` folder contains the **runtime infrastructure** that enables GPT-Ø to operate with revolutionary efficiency:

- **`neural_memory_runtime.py`**: Breakthrough 8GB RAM operation for 30-70B equivalent model through neural compression, hierarchical memory tiers, quantum-inspired states
- **`neural_model_manager.py`**: Dynamic model lifecycle management, resource allocation, performance monitoring
- **`cas_system.py`**: Cognitive Architecture Specification - constitutional AI safety, cognitive profiles, runtime adaptation
- **`cas_integration_bridge.py`**: Seamless integration between CAS runtime and core model components
- **`model_creation.py`**: Model identity/configuration system, no external model creation

### Self-Modifying Learning Paradigm

**NO PRE-TRAINING. NO IMPORTED WEIGHTS. NO STATIC PARAMETERS.**

GPT-Ø learns through:

1. **Interaction-Based Evolution**: Every conversation reshapes the model
2. **Real-Time Reasoning**: Dynamic weight computation based on context
3. **Memory Integration**: Neural memory runtime preserves and builds knowledge
4. **Bootstrap Learning**: Multiple output heads provide diverse world interaction capabilities
5. **Constitutional Evolution**: CAS system ensures safe, guided self-modification

### Development Environment Context

// Your environment is linux based, but my host operating system is Windows 11, with python 3.12 installed. Keep this in mind when making changes to the codebase.

// NO VIRTUAL ENVIRONMENT SETUP NEEDED - The user has their own kernel/backend orchestration handled externally
// NO BACKEND SETUP REQUIRED - This codebase is ONLY the model itself
// MODEL MUST: Initialize → Run → Interact via `run.py` immediately

### Component Integration Architecture

**Core Integration Flow:**

```terminal
run.py → gpt_model.py (main model) ← recursive_weights_core.py (weight computation)
                ↓
        tokenizer_mux.py + tokenizer_adapter.py (all 13+ modalities)
                ↓
        extra_output_heads/ (3 specialized neural outputs)
                ↓
        cas/ runtime system (memory, management, safety)
```

**Current Integration Status:**

- `tool_output_head.py`: ✅ Tied into system (needs import validation)
- `eyes_outputs.py`: ❌ NOT integrated into gpt_model.py, tokenizer_mux.py, or tokenizer_adapter.py
- `ears_outputs.py`: ❌ NOT integrated into gpt_model.py, tokenizer_mux.py, or tokenizer_adapter.py

**CAS Runtime Integration:**
The `/cas` folder provides the revolutionary runtime that enables 30-70B equivalent model performance on 8GB RAM:

- `neural_memory_runtime.py`: 5-tier memory hierarchy with neural/quantum compression
- `neural_model_manager.py`: Dynamic resource allocation and performance monitoring  
- `cas_system.py`: Constitutional AI safety framework and cognitive profiles
- `cas_integration_bridge.py`: Seamless CAS-to-core model integration
- `model_creation.py`: Identity/configuration system (NOT external model creation)

**Bootstrap Learning Strategy:**
Since there's NO pre-training, the multiple output heads serve as "world interface bootstraps":

- Tool head: Learns through system interaction and tool synthesis
- Eyes head: Learns through visual/surveillance data processing
- Ears head: Learns through spatial/audio domain processing
- Text generation: Learns through conversation and reasoning

### Critical Implementation Status & Gaps

**EMPTY/INCOMPLETE FILES REQUIRING IMPLEMENTATION:**

- `run.py`: ❌ EMPTY - Main entry point for model initialization and interaction
- `recursive_weights_core.py`: ❌ EMPTY - Core weight computation system
- `bayesian_config_orchestrator.py`: ❌ EMPTY - Real-time parameter optimization
- `tokenizer_adapter.py`: ❌ EMPTY - Multimodal tokenizer interface
- `eyes_outputs.py`: ❌ EMPTY - ISR output head (currently just imports)
- `ears_outputs.py`: ❌ PARTIAL - Spatial output head (has structure but incomplete)

**FILES WITH IMPLEMENTATION BUT NEEDING INTEGRATION:**

- `gpt_model.py`: ✅ HAS CONTENT - Core model structure exists but may need CAS integration
- `tokenizer_mux.py`: ✅ HAS CONTENT - Multimodal tokenizer framework exists
- `tool_output_head.py`: ✅ HAS CONTENT - Tool synthesis system exists
- `/cas/*`: ✅ HAVE CONTENT - Runtime systems exist but integration status unknown

### Your Task Focus

Your job is to:

1. **First update TODO.md with current system status**
2. **Update this JULES.md file frequently to maintain context**
3. **Focus PURELY on the model implementation - NO backend/infrastructure**
4. **Ensure ALL components are self-contained within the specified files**
5. **Complete integration of eyes_outputs.py and ears_outputs.py into the core system**
6. **Implement the EMPTY files to create a functioning self-modifying model**
7. **Validate that run.py can initialize, run, and interact with the complete system**

**CRITICAL:** This model must work RIGHT NOW via `python run.py` - no training, no external weights, purely interaction-based learning from the moment it starts.

## Agentic Development Workflow (Inspired by Jules Architecture)

### Task Lifecycle Pattern

1. **Context Analysis**: Full repository understanding before any changes
2. **Plan Generation**: Detailed multi-step execution plan with reasoning
3. **Human Review**: Mandatory approval checkpoint before execution
4. **Isolated Execution**: Changes made in secure, contained environment
5. **Validation**: Automated testing and quality checks
6. **Pull Request**: Structured delivery for final human review
7. **Feedback Integration**: Learning from outcomes for future improvements

### Autonomous Operation Principles

- **Transparency**: All intended actions clearly communicated
- **Controllability**: Human oversight at critical decision points
- **Isolation**: Safe execution environment preventing unintended consequences
- **Completeness**: Full task completion, not partial implementations
- **Learning**: Continuous improvement from interaction patterns

### Multi-Step Task Execution Framework

- Break complex requests into manageable, sequential steps
- Validate each step before proceeding to next
- Maintain context and dependencies across step boundaries
- Provide clear progress indicators and completion status
- Handle errors gracefully with rollback capabilities

## Development Guidelines

//DO NOT remove and/or delete or mess up my code meaning or functions. YOU ARE ONLY TO FULLY IMPLEMENT ANY MISSING COMPONENT OR TIE THEM TOGETHER. YOU WILL NOT DELETE CAPABILITIES IF UNIMPLEMENTED. YOU MUST FOLLOW THESE STRICT NON NEGOTIABLE CODING GUIDELINES

### Code Quality Standards

## Objectives

- Generate production-ready, secure, high-performance, maintainable code with zero stubs, zero dead code, and verifiable quality.
- Enforce measurable, automated quality gates at every change.

## Definition of Done (DoD)

A change is eligible to merge only if ALL of the following are true:

1. Functional completeness: No stubs, placeholders, pseudo-code, or commented-out code paths. No TODO/FIXME.
2. Tests:
   - Unit coverage ≥ 90% lines and ≥ 90% branches for changed files.
   - Mutation testing score ≥ 75% on changed files or rationale with risk acceptance by code owners.
   - Integration tests cover critical paths and error scenarios.
   - E2E tests updated if user-facing behavior changes.
   - Flaky test budget = 0; tests must be deterministic.
3. Static quality gates (CI-enforced):
   - Lint: 0 errors; warnings must be justified or fixed.
   - Type-check: 0 errors.
   - Complexity thresholds: cyclomatic ≤ 10 per function unless justified; functions > 50 LOC must be refactored or justified.
   - Duplication: no copy-paste beyond 3 lines without abstraction.
4. Security:
   - Secrets never hardcoded; use secure config/secret stores.
   - SAST: 0 high/critical findings; medium requires documented mitigation and code owner approval.
   - DAST (where applicable): 0 high/critical.
   - SCA: 0 vulnerable dependencies at high/critical; pinned versions or lockfiles required.
   - Threat model updated for new threat surfaces (STRIDE or equivalent) with mitigations.
   - Input validation, output encoding, and least privilege enforced. No direct string concatenation for queries; use parameterization.
5. Performance and reliability:
   - Performance SLOs defined for changed components with benchmarks or load tests for hot paths. Regressions > 2% require approval and tracking.
   - Memory safety and bounded resource usage; no unbounded growth or leaks.
   - Concurrency safety: no data races; proper synchronization or immutability.
   - Timeouts, retries with backoff, and circuit breakers for network/IO.
6. Observability and operations:
   - Structured logs with correlation IDs; no sensitive data in logs.
   - Metrics (latency, errors, resource) and health checks for services.
   - Tracing added for critical spans.
   - Feature flags for risky changes; documented rollout/rollback.
7. Accessibility and UX (for UI):
   - WCAG 2.1 AA: keyboard navigability, contrast, ARIA semantics, focus management.
   - Screen reader support validated.
8. Privacy and compliance:
   - Data classification applied; PII/PHI handled per policy.
   - Data minimization, purpose limitation, retention documented.
   - Redaction for logs/exports; encryption in transit and at rest where applicable.
9. Documentation:
   - Public APIs/classes/functions have docstrings with parameters, returns, errors, examples.
   - Architectural decision record (ADR) for non-trivial changes.
   - Update READMEs, runbooks, and migration notes.
10. Code review and governance:

- At least 2 approving reviews from code owners for critical areas; 1 for others.
- No self-approval; no force merges.
- Commit messages: imperative, reference issue/ADR, describe rationale and impact.
- Trunk-based development with short-lived branches; CI green required.

## 11. Backwards compatibility:

- Avoid breaking changes; if unavoidable, provide migrations, deprecation schedule, and clear release notes.

## 12. Internationalization (if applicable):

- No hardcoded user-facing strings; use i18n framework and pluralization rules.

## Mandatory Practices

- Full implementation only; partial solutions are rejected.
- Defensive programming: validate inputs, handle all error paths, and fail fast with clear messages.
- Idempotency for mutation endpoints and jobs where applicable.
- Resource cleanup and cancellation support.
- Use established patterns: dependency injection, single responsibility, clear boundaries.
- Interfaces and contracts are explicit; avoid hidden side-effects.

## Tooling and Automation (CI Quality Gates)

- Linting and formatting: enforced via CI. Config checked into repo.
- Type checking: enforced with strict mode where available.
- SAST: run on every PR (e.g., CodeQL/Bandit/ESLint security).
- DAST: run on protected branches or preview envs for services with HTTP surfaces.
- SCA and license compliance: fail on incompatible licenses or high/critical CVEs; use allowlists/overrides with owner sign-off.
- IaC scanning for Terraform/K8s/Cloud configs; 0 criticals.
- Container checks: minimal base images, non-root user, pinned tags/digests, vulnerability scan gates.
- Mutation testing for critical libraries/services on changed files.
- Test flake detection and quarantine require issue creation and fix before merge.

## Performance and Load Testing

- Provide micro-benchmarks for algorithms and hot paths.
- For services: load tests with representative traffic; define throughput/latency/error budgets. Document baseline and delta.
- Use profiling to justify algorithmic choices when complexity > O(n log n) or large constants.

## Error Handling and Resilience

- No silent failures. Log at appropriate levels with actionable context.
- Wrap external calls; implement retries, timeouts, and jittered backoff.
- Graceful degradation and circuit breakers under partial outages.
- Dead-letter queues or retries for async processing with observability.

## Security Best Practices (Non-Exhaustive)

- Use parameterized queries; no dynamic SQL string concatenation.
- Validate and sanitize all external inputs; whitelist preferred over blacklist.
- Escape/encode outputs to prevent XSS/HTML injection.
- CSRF protection for state-changing HTTP endpoints.
- Strong password policies; modern KDFs (e.g., Argon2/bcrypt/scrypt) with salt and appropriate cost.
- JWT/session hardening: short TTLs, rotation, audience/issuer checks.
- Enforce least privilege IAM and scoped tokens.
- Regular key rotation; secrets from env/secret manager, not from code or VCS.
- Logging excludes secrets, tokens, credentials, and sensitive identifiers.

## Maintainability and Readability

- Small, cohesive modules and functions with clear names.
- Avoid deep inheritance; prefer composition.
- Public APIs stable and documented; internal details hidden.
- Comments explain why, not what; code should be self-explanatory.
- No premature optimization; measure and document when optimizing.

## Acceptance Checklist (must be met before merge)

- [ ] No partial implementations; no TODO/FIXME.
- [ ] All inputs validated; all error paths handled; resources cleaned up.
- [ ] Style/lint/type checks pass with 0 errors.
- [ ] Line and branch coverage ≥ 90% for changed files; mutation ≥ 75% or approved exception.
- [ ] Integration/E2E tests updated; no flakiness.
- [ ] SAST/SCA/IaC/container scans: 0 high/critical; mediums documented or fixed.
- [ ] Performance baselines present; no >2% regression without approval.
- [ ] Observability (logs/metrics/traces/health) implemented.
- [ ] ADRs/docs/runbooks updated.
- [ ] Accessibility (WCAG 2.1 AA) verified where applicable.
- [ ] Privacy/compliance reviewed; no sensitive data leakage.
- [ ] Reviews completed per code owners; CI green.
- [ ] Rollout/rollback plan documented; feature flags where needed.

## Immediate Escalation

If any requirement cannot be met:

1. Halt coding.
2. Document the blocker with evidence.
3. Notify code owners and request guidance or a risk acceptance decision.

Non-compliance will result in immediate rejection of code changes.

### Implementation Priorities

1. Fix TODO, README.md, Update JULES.md for your task and more project context.
2. Once done with analzying and TODO, you are to follow these documents and begin fully implementing the fixes to the codebase.
3. YOU CAN NOT STRAY FROM CODING GUIDELINES OR PRIORITIES.
