## Project Overview

This is an AI training infrastructure library providing unified abstractions for policies (models), domains (problem sets), graders (reward functions), and trainers (training strategies).

**Key files to understand:**
- `README.md` - Full project documentation
- `core/schema.py` - Base Config class used throughout
- `core/policy/schema.py` - Policy interface and sample types
- `core/domain/schema.py` - Domain interface and problem types
- `core/grader/schema.py` - Grader interface
- `core/trainer/schema.py` - Trainer interface

**Naming conventions:**
- Files and symbols should be extensible, informative, and self-explanatory
- Use descriptive names that indicate purpose and scope

## Project Structure

This is an infrastructure-only repository. There are no scripts, algorithms, or evaluation frameworks here - only the core abstractions and implementations.

**Key directories:**
- `core/domain/` - Problem domain implementations
- `core/policy/` - Model interface implementations
- `core/grader/` - Reward/grading implementations
- `core/trainer/` - Training strategy implementations
- `utils/` - Utility functions
- `lib/safety_tooling/` - Production API inference library
- `data/config/` - Training configuration files (DeepSpeed, Accelerate)
- `data/questions/` - Domain-specific datasets

## Development Strategy

### Code Reuse
- Extend or fix existing classes/methods rather than duplicating functionality
- Policy, Domain, Grader, and Trainer all have base classes - inherit from them
- Use composition over inheritance where appropriate (e.g., wrapping graders)

### Error Handling
- Don't catch exceptions that indicate bugs - let them fail fast
- Use assertions for invariants that should never fail
- NEVER use mock data or suppress unexpected errors
- Validate inputs at API boundaries, fail clearly on invalid data

### Testing and Validation
- Test with real data, not mocks
- For analysis/debugging, include raw outputs (dataframe summaries, full error traces)
- Run end-to-end tests and inspect outputs manually
- Iterate until everything works correctly

### Code Organization
- Keep logic unified - avoid duplication
- Extract common patterns into utilities
- Use type hints for clarity
- Document complex logic with comments

## Development Conventions

### Import Path Management
Always add this at the top of your scripts:
```python
import utils.path_utils  # Fixes import paths
```

Run scripts as modules from project root:
```bash
python -m your.module.path
# NOT: python your/module/path.py
```

### Async/Sync Interop
Run async functions from sync context:
```python
from utils.async_utils import run_coroutine

result = run_coroutine(async_function(*args, **kwargs))
```

### Code Quality
After making changes:
1. **Clean up** - Remove debugging code, unused imports
2. **Lint** - `ruff check --fix .`
3. **Format** - `black <changed_file> --workers=1` (only changed files!)
4. **Test** - Run relevant tests to ensure nothing broke

### Working with the Abstractions

#### Creating a New Domain
1. Inherit from `ProblemDomain` (core/domain/schema.py:128)
2. Define your questions as `BinaryProblem` or `OpenEndedProblem`
3. Implement `make_questions_splits()` if needed
4. Override `postprocess_sample()` and `preprocess_samples()` if needed

#### Creating a New Grader
1. Inherit from `Grader` (core/grader/schema.py:17)
2. Implement `grade_async(sample, item)` - return float score
3. Implement `to_openai_spec()` for OpenAI RL API compatibility
4. Optionally implement `validate_problem()` and `transform_dataset()`

#### Creating a New Trainer
1. Inherit from `Trainer` (core/trainer/schema.py:62)
2. Implement `train_async(policy, trajectory_score_files, reasoning_mode, **kwargs)`
3. Use helper methods: `load_trajectory_scores()`, `select_top_trajectories()`, `build_metadata()`
4. Training is always out-of-place - return new Policy instance

#### Creating a New Policy
1. Inherit from `Policy` (core/policy/schema.py:51)
2. Implement `infer_single_async()` for inference
3. Optionally implement `logprobs_single_async()` for log probability support
4. Optionally implement `train_sft_async()`, `train_rl_async()` for training
5. Use `deep_copy()` utility when creating trained variants