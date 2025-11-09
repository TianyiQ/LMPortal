# AI Training Infrastructure

## Overview

This repository provides a unified infrastructure for language model training and inference. It defines clean abstractions for **policies** (models), **domains** (problem sets), **graders** (reward functions), and **trainers** (training strategies), enabling flexible experimentation with different combinations of these components.

Key features:
- **Unified Policy Interface**: Work with API models, local models, batch APIs, Claude Code agents, and even humans through the same interface
- **Flexible Training**: Support for SFT (via OpenAI/TogetherAI API or local), RL (via OpenAI/TogetherAI API or local), and few-shot learning
- **Domain Abstractions**: Structured problem definitions for forecasting, research Q&A, and conceptual reasoning
- **Grader Framework**: Python-based and model-based graders for automatic reward computation
- **Production-Ready**: Includes the `safety_tooling` library for robust API inference with caching, retry logic, and batch processing

## Installation

```bash
# Install the safety_tooling library for API inference
uv pip install -e lib/safety_tooling

# Install the main package
uv pip install -e .
```

## Usage Examples

### Example 1: Basic Policy Inference

```python
from utils.policy_utils import create_policy_from_string

# Create a policy (automatically detects provider)
policy = create_policy_from_string("o4-mini")

# Single inference
response = policy.infer_single("What is the capital of France?")
print(response)

# Batch inference
responses = policy.infer_batch([
    "What is 2+2?",
    "Name three programming languages.",
])
```

### Example 2: Working with Domains

```python
from core.domain.conceptual import ConceptualDomain

# Load domain
domain = ConceptualDomain()

# Sample problems
problems = domain.sample_problems(n=5, split="train")

for problem in problems:
    print(f"Q: {problem.question}")
    if hasattr(problem, "correct_option"):
        print(f"Answer: {problem.options[problem.correct_option]}")
```

### Example 3: Supervised Fine-Tuning

```python
from utils.policy_utils import create_policy_from_string
from core.policy.schema import SingleSample
from core.trainer.sft import SFTTrainer, SFTConfig

# Prepare training data
samples = [
    SingleSample(
        history=[{"role": "user", "content": "What is 2+2?"}],
        output="4",
    ),
    # ... more samples
]

# Create trainer
config = SFTConfig(
    num_epochs=2,
    learning_rate=1e-5,
    validation_strategy="train" # split from training set
)
trainer = SFTTrainer(config)

# Train (creates new policy, doesn't modify original)
base_policy = create_policy_from_string("gpt-4o")
trained_policy = trainer.train(
    policy=base_policy,
    trajectory_score_files=["path/to/scored_trajectories.json"]
)
```

### Example 4: Reinforcement Learning with Graders

```python
from core.domain.forecasting import ForecastingDomain
from core.trainer.rl import RLTrainer, RLConfig
from core.grader.python_brier import PythonBrierGrader

# Setup
domain = ForecastingDomain()
problems = domain.sample_problems(n=100, split="train")

# Create grader and trainer
grader = PythonBrierGrader()
config = RLConfig(num_epochs=3, learning_rate=1e-6, kl_coef=0.1)
trainer = RLTrainer(config)

# Train with RL
base_policy = create_policy_from_string("o4-mini")
trained_policy = trainer.train(
    policy=base_policy,
    problem_list=problems,
    grader=grader
)
```

### Example 5: Local Model Training with Multi-GPU

```python
from core.policy.localmodel import LocalModel
from core.trainer.sft import SFTTrainer

# Create local model (automatically uses all available GPUs)
model = LocalModel("meta-llama/Llama-3.2-1B-Instruct")

# Train with DeepSpeed ZeRO-2 (automatic)
trainer = SFTTrainer(SFTConfig(num_epochs=2))
trained_model = await trainer.train_async(
    policy=model,
    trajectory_score_files=["trajectories.json"]
)
```

## Core Components

The codebase is organized into four main abstraction layers:

### 1. Domains (`core/domain/`)

Domains define problem sets with structured questions and optional ground truth. Base class: `ProblemDomain` (core/domain/schema.py:128)

**Problem Types:**
- `BinaryProblem` - Questions with Yes/No options and optional ground truth (core/domain/schema.py:21)
- `OpenEndedProblem` - Questions without predefined answers (core/domain/schema.py:116)

**Available Domains:**
- **`forecasting.py`** - Binary prediction questions (requires fetching data)
- **`research.py`** - Research Q&A with easy/hard answer pairs
- **`conceptual.py`** - 31 conceptual/philosophical questions
- **`intellectual.py`** - Intellectual reasoning questions
- **`openreview.py`** - Academic paper review tasks
- **`cmvbinary.py`** / **`cmvfreeform.py`** - ChangeMyView opinion evaluation

**Key Methods:**
- `sample_problems(n, split)` - Sample without replacement from train/test splits
- `make_questions_splits(train_size)` - Create train/test splits

### 2. Policies (`core/policy/`)

Policies are unified interfaces for language models. Base class: `Policy` (core/policy/schema.py:51)

**Available Implementations:**
- **`apimodel.py`** - Standard API-based models (OpenAI, Anthropic, DeepSeek, etc.)
- **`raymodel.py`** - Ray-parallelized API calls for high-throughput workloads (>100k tokens/s)
- **`batchmodel.py`** - Provider batch APIs for 50% cost reduction (24-48hr latency)
- **`localmodel.py`** - Local deployment with SGLang backend, supports logprobs and training
- **`human.py`** - CLI-based human-in-the-loop policy
- **`claudecode.py`** - Claude Code agent integration

**Key Methods:**
- `infer_single(history)` / `infer_batch(histories)` - Generate completions
- `logprobs_single(dialogue)` / `logprobs_batch(dialogues)` - Get log probabilities (local models only)
- `train_sft(samples)` / `train_rl(samples, grader)` - Train the model (out-of-place, returns new policy)
- `add_few_shot_examples(examples)` - Create policy with few-shot context (out-of-place)
- `embed(texts)` - Generate embeddings (where supported)

**Sample Types** (core/policy/schema.py:24-48):
- `SingleSample` - History + output for SFT
- `PairedSample` - History + winning/losing outputs for DPO
- `EvaluatedSample` - History + output + reward for RL

### 3. Graders (`core/grader/`)

Graders compute rewards for RL training or evaluation scores. Base class: `Grader` (core/grader/schema.py:17)

**Available Implementations:**
- **`python_brier.py`** - Extracts `\finalBeliefProb{X}` patterns and computes Brier scores
- **`model_brier.py`** - Uses LLMs to extract beliefs, then computes Brier scores
- **`model_agreement.py`** - Uses LLMs to grade agreement/correctness
- **`python_grader.py`** - Custom Python grading logic (can run on OpenAI servers for RL)
- **`model_grader.py`** - Custom model-based grading with prompts

**Key Methods:**
- `grade(sample, item)` - Compute reward/score for a sample
- `to_openai_spec()` - Convert to OpenAI RL API format
- `validate_problem(problem)` - Check if problem is suitable for this grader
- `transform_dataset(problems)` - Add instructions or format problems

**Factory Functions:**
- `create_grader_from_spec(spec)` - Create grader from dict/string/callable
- `create_grader_from_env()` - Create grader from environment variables (`GRADER_TYPE`, `GRADER_MODEL`)

### 4. Trainers (`core/trainer/`)

Trainers orchestrate the training process. Base class: `Trainer` (core/trainer/schema.py:62)

**Available Implementations:**
- **`sft.py`** - Supervised fine-tuning on top-scoring trajectories
  - Supports OpenAI/Together APIs and local training (TRL + DeepSpeed)
  - Automatic validation set creation (none/train/gt strategies)
  - WandB logging support

- **`rl.py`** - Reinforcement learning with custom graders
  - Supports OpenAI RL API and local training (TRL GRPO)
  - Works with any `Grader` implementation
  - Configurable KL penalty and reward shaping

- **`fewshot.py`** - Few-shot in-context learning
  - Selects top trajectories as examples
  - Creates new policy with prepended context (out-of-place)

**Key Methods:**
- `train(policy, trajectory_score_files, reasoning_mode)` - Main training entry point
- `load_trajectory_scores(filepath)` - Load trajectory-score pairs
- `select_top_trajectories(pairs, top_percentage, top_count)` - Filter by score
- `build_metadata(policy, files)` - Create training metadata

**Configuration** (core/trainer/schema.py:24):
- `validation_strategy` - "none", "train" (split from training), or "gt" (ground truth filtered)
- `lora_rank` - LoRA rank (0 for full-parameter training)
- Set via environment variables: `VALIDATION_STRATEGY`, `LORA_RANK`

## Codebase Structure

```
.
├── core/                    # Core abstractions
│   ├── domain/             # Problem domains (forecasting, research, etc.)
│   ├── grader/             # Reward/grading functions
│   ├── policy/             # Model interfaces (API, local, batch, human)
│   ├── trainer/            # Training strategies (SFT, RL, few-shot)
│   └── schema.py           # Base Config class
│
├── utils/                   # Utility functions
│   ├── policy_utils.py     # Policy creation and management
│   ├── io_utils.py         # I/O operations and JSON handling
│   ├── async_utils.py      # Async helpers (run_coroutine)
│   ├── path_utils.py       # Import path fixes
│   ├── stats_utils.py      # Statistical analysis tools
│   └── templates/          # Prompt templates
│
├── lib/safety_tooling/      # API inference library (see lib/safety_tooling/README.md)
│   ├── safetytooling/apis/inference/  # API clients (OpenAI, Anthropic, etc.)
│   ├── safetytooling/data_models/     # Data models for requests/responses
│   └── safetytooling/utils/           # Caching, retry logic, utilities
│
└── data/                    # Data and configuration
    ├── config/             # Training configs (DeepSpeed, Accelerate)
    └── questions/          # Domain-specific question datasets
```

## Environment Variables

### API Keys

Required for API-based policies:
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `TOGETHER_API_KEY` - Together AI API key
- `DEEPSEEK_API_KEY` - DeepSeek API key
- `GOOGLE_API_KEY` - Google (Gemini) API key
- `HUGGINGFACE_API_KEY` - HuggingFace API key
- `OPENROUTER_API_KEY` - OpenRouter API key
- `WANDB_API_KEY` - Weights & Biases logging (optional)

### Training Configuration

Used by trainer implementations:
- `VALIDATION_STRATEGY` - Validation set strategy: "none", "train", "gt" (default: "none")
- `LORA_RANK` - LoRA rank for parameter-efficient training (default: 0, full-parameter)
- `TRAINED_POLICY_NAME_PATTERN` - Naming pattern for trained models (supports placeholders)
- `GRADER_TYPE` - Grader type: "python_brier", "model_brier", "model_agreement", "model"
- `GRADER_MODEL` - Model name for model-based graders (default: "o4-mini-2025-04-16")
- `GRADER_SPEC` - Full grader specification (JSON string)

### Performance and Execution

- `USE_RAY` - Enable Ray for parallel API calls (default: true)
- `USE_OPENROUTER` - Use OpenRouter for model routing (requires USE_RAY=true)
- `USE_BATCH` - Use provider batch APIs for cost savings (requires USE_RAY=false)
- `MAX_WORKERS` - Maximum Ray workers
- `LOCALMODEL_MAX_CONCURRENT` - Max concurrent local model instances
- `FORCE_SINGLE_GPU` - Force single-GPU usage (debugging)
- `DISABLE_DEEPSPEED` - Disable DeepSpeed, use regular DDP
- `NO_RETRY` - Disable retry mechanism for API calls

### Sampling and Data

- `DEFAULT_SPLIT` - Default data split: "train" or "test" (default: "train")
- `TEMPERATURE` - Model temperature (default: 0.25)
- `PRESENCE_PENALTY` - Presence penalty (default: 0.0)


## Advanced Features

### Multi-GPU Training with DeepSpeed

LocalModel supports distributed training across multiple GPUs automatically:

```bash
# Automatic multi-GPU detection
python your_training_script.py

# Force single GPU (debugging)
FORCE_SINGLE_GPU=1 python your_training_script.py

# Use Accelerate launcher for explicit control
accelerate launch --config_file data/config/accelerate_config_1node_4gpu.yaml your_script.py
```

DeepSpeed ZeRO-2 is automatically used when multiple GPUs are detected. Configuration files in `data/config/`:
- `deepspeed_zero2.json` - ZeRO Stage 2 (recommended)
- `deepspeed_zero3.json` - ZeRO Stage 3 (very large models)
- `accelerate_config_1node_{N}gpu.yaml` - Accelerate configs for N GPUs

### Batch APIs for Cost Savings

Use `BatchModel` for 50% cost reduction (24-48hr latency):

```python
from core.policy.batchmodel import BatchModel

policy = BatchModel("gpt-4o-mini")
# Same interface as other policies, but uses batch API
```

### Ray-based Parallelization

For high-throughput workloads (>100k tokens/s):

```python
from core.policy.raymodel import RayModel

policy = RayModel("gpt-4o-mini")
# Automatically parallelizes API calls across workers
```

## Safety Tooling Library

The `lib/safety_tooling` package provides production-ready API inference with:
- **Caching**: Redis-based caching for cost savings
- **Retry Logic**: Exponential backoff with configurable retries
- **Rate Limiting**: Automatic rate limit handling
- **Batch Processing**: Efficient batch API support
- **Multiple Providers**: OpenAI, Anthropic, DeepSeek, Google, Together, HuggingFace

See `lib/safety_tooling/README.md` for detailed documentation.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
ruff check --fix .
black . --workers=1
```

### Project Structure Best Practices

From `CLAUDE.md`:
- Use `import utils.path_utils` at the top of scripts
- Run modules with `python -m module.path` from project root
- For async from sync context: `from utils.async_utils import run_coroutine`

## License

MIT

## Citation

If you use this codebase, please cite:

```bibtex
@software{truthseekinggym2025,
  title = {AI Training Infrastructure},
  author = {Research Team},
  year = {2025},
  url = {https://github.com/yourusername/truthseekinggym}
}
```
