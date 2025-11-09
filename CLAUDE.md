# Development Agent Guide

## Project Context

This is an **infrastructure library** for AI training. You're working with:
- **Core abstractions**: Policy (models), Domain (problems), Grader (rewards), Trainer (training)
- **No scripts, algorithms, or evaluation code** - only the foundational abstractions
- Focus on clean interfaces, extensibility, and production quality

## Quick Orientation

### Essential Reading
1. `README.md` - Project overview and API documentation
2. `core/schema.py` - Base Config class
3. `core/*/schema.py` - Interface contracts for each component

### Directory Structure
```
core/
  domain/    - Problem sets (forecasting, research Q&A, etc.)
  policy/    - Model interfaces (API, local, batch, human)
  grader/    - Reward functions for RL
  trainer/   - Training strategies (SFT, RL, few-shot)
utils/       - Helpers (async, I/O, policy creation)
lib/safety_tooling/ - Production API inference library
data/
  config/   - Training configs (DeepSpeed, Accelerate)
  questions/ - Domain datasets
```

### Running Commands

```bash
# Always run as module from project root
python -m your.module.path

# NOT: python your/module/path.py
```

Also, to avoid issues with multiple event loops, use `utils.async_utils.run_coroutine` to run async functions from sync context.

```python
from utils.async_utils import run_coroutine

# Call async from sync context
result = run_coroutine(async_function(*args))
```

## Coding Standards

### Design Principles
- **Extend, don't duplicate** - Inherit from base classes (Policy, Domain, Grader, Trainer)
- **Fail fast** - Don't catch exceptions that indicate bugs
- **Real data only** - NEVER use mocks or suppress errors
- **Out-of-place operations** - Training returns new instances, doesn't mutate

### Code Quality Checklist
- [ ] Type hints on all public APIs
- [ ] Docstrings for classes and non-trivial methods
- [ ] Assertions for invariants
- [ ] No dead code or debug prints
- [ ] Linted (`ruff`) and formatted (`black`)

### Common Patterns

#### Creating a New Domain
```python
from core.domain.schema import ProblemDomain, BinaryProblem

class MyDomain(ProblemDomain):
    def __init__(self):
        super().__init__()
        self.questions_all = [...]  # Load your questions
        self.make_questions_splits(train_size=0.8)
```

#### Creating a New Grader
```python
from core.grader.schema import Grader

class MyGrader(Grader):
    async def grade_async(self, sample, item=None):
        # Return float score
        return score

    def to_openai_spec(self):
        # Return OpenAI RL API format
        return {"type": "...", ...}
```

#### Creating a New Trainer
```python
from core.trainer.schema import Trainer

class MyTrainer(Trainer):
    async def train_async(self, policy, trajectory_score_files, ...):
        # Load data
        pairs = self.load_trajectory_scores(trajectory_score_files[0])
        trajs = self.select_top_trajectories(pairs, top_percentage=0.1)

        # Convert and train (out-of-place!)
        samples = convert_to_samples(trajs)
        trained = await policy.train_sft_async(samples)
        return trained
```

## Testing Strategy

### Test with Real Data
```python
# Good
domain = ForecastingDomain()
problems = domain.sample_problems(n=5)

# Bad - NEVER
problems = [MockProblem(), MockProblem()]
```

### Validate Outputs
- Run end-to-end tests manually
- Inspect actual outputs (print dataframes, check files)
- Iterate until correct

### Debug Effectively
```python
# Include full context in errors
logger.urgent(f"Failed to process {item}: {error}\n{traceback.format_exc()}")

# NOT: just swallow exceptions
```

## Communication

### Progress Updates
- Keep short and action-focused
- State what you're doing and why
- For long operations, give success criteria

### Final Summary
Include:
1. **What changed** - High-level description
2. **Files modified** - List of changed files
3. **How to verify** - Exact command to test
4. **Known issues** - Any caveats or TODOs

Example:
```
Updated ForecastingDomain to support new data format.

Files changed:
- core/domain/forecasting.py
- tests/test_forecasting.py

Verify: python -m pytest tests/test_forecasting.py

Note: Requires fetching new data first with the domain's data fetching method.
```

## Common Pitfalls

❌ **Don't**:
- Catch exceptions that shouldn't happen
- Use mock data in production code
- Mutate policies/configs in-place
- Run files directly (`python file.py`)
- Format unchanged files

✅ **Do**:
- Let bugs fail fast with clear errors
- Test with real domains/problems
- Return new instances from training
- Run as modules (`python -m module`)
- Format only what you changed

## Quick Reference

### Factory Functions
```python
from utils.policy_utils import create_policy_from_string
from core.grader.schema import create_grader_from_env

policy = create_policy_from_string("o4-mini")
grader = create_grader_from_env()  # Uses GRADER_TYPE env var
```

### Environment Variables
- API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
- Training: `VALIDATION_STRATEGY`, `LORA_RANK`, etc.
- Performance: `USE_RAY`, `FORCE_SINGLE_GPU`, etc.

### File References
- Policy interface: `core/policy/schema.py:51`
- Domain interface: `core/domain/schema.py:128`
- Grader interface: `core/grader/schema.py:17`
- Trainer interface: `core/trainer/schema.py:62`