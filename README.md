# AI Training Infrastructure

## Overview

This repository provides a unified infrastructure for language model training and inference. It defines clean abstractions for **policies** (models), **domains** (problem sets), **graders** (reward functions), and **trainers** (training strategies), enabling flexible experimentation with different combinations of these components.

Key features:
- **Unified Policy Interface**: Work with API models, local models, batch APIs, Claude Code agents, and even humans through the same interface
- **Flexible Inference**: The new `infer()` and `infer_many()` methods accept multiple input types (histories, Samples, Problems, or Domains) and return appropriate output types
- **Fully Parallelized**: The pipeline is fully asynchronous and parallelized, achieving maximum concurrency for both inference and training. Optional support for Ray to increase multi-core CPU utilization
- **Flexible Training**: Support for SFT (via OpenAI/TogetherAI API or local), RL (via OpenAI API or local), and few-shot learning
- **Domain Abstractions**: Problem domain interface, with predefined implementations for forecasting, research Q&A, conceptual reasoning, intellectual reasoning, OpenReview, and ChangeMyView opinion evaluation tasks
- **Grader Framework**: Grader interface (Python-based/LLM-based), with predefined implementations for Brier score and agreement score

## Installation

```bash
# Install the safety_tooling library for API inference
uv pip install -e lib/safety_tooling

# Install the main package
uv pip install -e .

# Enter your API keys
cp lib/safety_tooling/.env.example lib/safety_tooling/.env
vi lib/safety_tooling/.env
```

## Recommended Configuration (optional)

For optimal performance, set these optional environment variables:

```bash
export USE_RAY=1          # Enable Ray for parallel API calls and multi-core utilization
export USE_OPENROUTER=1   # Use OpenRouter for high-throughput model routing
```

## Usage Examples

### Example 1: Basic Flexible Inference

The `infer()` method is the recommended way to do inference - it accepts multiple input types and returns appropriate outputs:

```python
from utils.policy_utils import create_policy_from_string

# Create a policy (automatically detects provider)
policy = create_policy_from_string("o4-mini")

# Simple string inference
response = policy.infer("What is the capital of France?")
print(response)  # Returns: str

# Or with history
response = policy.infer([
    {"role": "user", "content": "What is 2+2?"}
])
print(response)  # Returns: str
```

### Example 2: Inference with Problems and Domains

The flexible `infer()` method can directly work with Problems and Domains:

```python
from core.domain.conceptual import Conceptual
from utils.policy_utils import create_policy_from_string

policy = create_policy_from_string("o4-mini")
domain = Conceptual()

# Infer from a single problem
problem = domain.sample_problems(n=1)[0]
result = policy.infer(problem.to_sample())
print(f"Question: {result.history[0]['content']}")
print(f"Answer: {result.output}")  # Returns: SingleSample

# Infer directly from domain (samples 1 problem automatically)
result = policy.infer(domain)
print(result)  # Returns: SingleSample
```

### Example 3: Batch Flexible Inference

The `infer_many()` method handles batch inference with flexible input types:

```python
from utils.policy_utils import create_policy_from_string
from core.domain.conceptual import Conceptual

policy = create_policy_from_string("o4-mini")
domain = Conceptual()

# Batch inference from multiple problems
problems = domain.sample_problems(n=3)
samples = [p.to_sample() for p in problems]
results = policy.infer_many(samples)
for result in results:
    print(f"Q: {result.history[0]['content']}")
    print(f"A: {result.output}")
# Returns: list[SingleSample]

# Or directly from domain with count
results = policy.infer_many((domain, 5))  # Sample 5 problems from domain
print(f"Generated {len(results)} responses")  # Returns: list[SingleSample]
```

### Example 4: Working with Domains

```python
from core.domain.conceptual import Conceptual

# Load domain
domain = Conceptual()

# Sample problems
problems = domain.sample_problems(n=5, split="train")

for problem in problems:
    print(f"Q: {problem.question}")
    if hasattr(problem, "correct_option"):
        print(f"Answer: {problem.options[problem.correct_option]}")

    # Convert problem to Sample for inference
    sample = problem.to_sample()
    print(f"Sample history: {sample.history}")
```

### Example 5: Human-AI Dialogue

Create interactive dialogues between human and AI policies:

```python
from core.policy.human import Human
from utils.policy_utils import create_policy_from_string

# Create policies
human = Human()
ai = create_policy_from_string("o4-mini")

# Start dialogue
history = []
for turn in range(3):
    # Human turn
    human_msg = human.infer_from_history(history)
    history.append({"role": "user", "content": human_msg})
    print(f"Human: {human_msg}")

    # AI turn
    ai_msg = ai.infer_from_history(history)
    history.append({"role": "assistant", "content": ai_msg})
    print(f"AI: {ai_msg}")
```

### Example 6: Claude Code Agent Inference

Use Claude Code agents for complex reasoning tasks:

```python
from core.policy.claudecode import ClaudeCode

# Create Claude Code agent policy
agent = ClaudeCode()

# Infer with code execution capabilities
result = agent.infer("Write a Python function to calculate fibonacci numbers and test it with n=10")
print(f"Agent response: {result}")
```

### Example 7: Supervised Fine-Tuning

SFT trainer accepts `list[SingleSample]` directly.

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
    SingleSample(
        history=[{"role": "user", "content": "What is the capital of France?"}],
        output="Paris",
    ),
    # ... more samples
]

# Create trainer
config = SFTConfig(
    num_epochs=2,
    learning_rate=1e-5,
    validation_strategy="train"  # split from training set
)
trainer = SFTTrainer(config)

# Train (creates new policy, doesn't modify original)
base_policy = create_policy_from_string("gpt-4o")
trained_policy = trainer.train(
    policy=base_policy,
    samples=samples
)
```

### Example 8: Few-Shot Learning

Few-shot trainer also accepts `list[SingleSample]`.

```python
from utils.policy_utils import create_policy_from_string
from core.policy.schema import SingleSample
from core.trainer.fewshot import FewShotTrainer

# Prepare few-shot examples
examples = [
    SingleSample(
        history=[{"role": "user", "content": "Translate to French: Hello"}],
        output="Bonjour",
    ),
    SingleSample(
        history=[{"role": "user", "content": "Translate to French: Goodbye"}],
        output="Au revoir",
    ),
]

# Create policy with few-shot examples
trainer = FewShotTrainer()
base_policy = create_policy_from_string("o4-mini")
fewshot_policy = trainer.train(
    policy=base_policy,
    samples=examples
)

# Now use the policy with in-context examples
response = fewshot_policy.infer("Translate to French: Thank you")
print(response)
```

### Example 9: Reinforcement Learning with Graders

```python
from core.domain.forecasting import Forecasting
from core.trainer.rl import RLTrainer, RLConfig
from core.grader.python_brier import PythonBrierGrader

# Setup
domain = Forecasting()
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

### Example 10: End-to-End Workflow

Complete workflow from domain to inference to training:

```python
from core.domain.conceptual import Conceptual
from utils.policy_utils import create_policy_from_string
from core.trainer.sft import SFTTrainer, SFTConfig

# 1. Load domain and sample problems
domain = Conceptual()
problems = domain.sample_problems(n=10, split="train")

# 2. Generate responses with base policy
policy = create_policy_from_string("o4-mini")
samples = [p.to_sample() for p in problems]
results = policy.infer_many(samples)

# 3. Use results as training data
trainer = SFTTrainer(SFTConfig(num_epochs=1))
trained_policy = trainer.train(policy=policy, samples=results)

# 4. Test trained policy
test_problem = domain.sample_problems(n=1, split="test")[0]
response = trained_policy.infer(test_problem.to_sample())
print(f"Q: {response.history[0]['content']}")
print(f"A: {response.output}")
```

### Example 11: Async Inference and Training Across Multiple Domains

Run inference and training on multiple domains in parallel.

```python
import asyncio
from core.domain.conceptual import Conceptual
from core.domain.forecasting import Forecasting
from utils.policy_utils import create_policy_from_string
from core.trainer.sft import SFTTrainer

async def process_domain(domain, policy, trainer):
    """Infer and train on a single domain"""
    # Generate training data
    problems = domain.sample_problems(n=5, split="train")
    samples = [p.to_sample() for p in problems]
    results = await asyncio.gather(*[policy.infer_async(s) for s in samples])

    # Train and return
    return await trainer.train_async(policy=policy, samples=results)

async def main():
    policy = create_policy_from_string("o4-mini")
    trainer = SFTTrainer()

    # Process multiple domains in parallel
    domains = [Conceptual(), Forecasting()]
    trained_policies = await asyncio.gather(
        *[process_domain(d, policy, trainer) for d in domains]
    )

    print(f"Trained {len(trained_policies)} policies in parallel")

asyncio.run(main())
```

Everything else in this library is also asynchronous, and the snippet above serves only as an example.

### Example 12: Local Model Training with Multi-GPU

```python
from core.policy.localmodel import LocalModel
from core.trainer.sft import SFTTrainer, SFTConfig
from core.policy.schema import SingleSample

# Create local model (automatically uses all available GPUs)
model = LocalModel("meta-llama/Llama-3.2-1B-Instruct")

# Prepare samples
samples = [
    SingleSample(
        history=[{"role": "user", "content": "Hello"}],
        output="Hi there!",
    ),
    # ... more samples
]

# Train with DeepSpeed ZeRO-2 (automatic)
trainer = SFTTrainer(SFTConfig(num_epochs=2))
trained_model = await trainer.train_async(
    policy=model,
    samples=samples
)
```

## Core Components

The codebase is organized into four main abstraction layers:

### 1. Domains (`core/domain/`)

Domains define problem sets with structured questions and optional ground truth. Base class: `ProblemDomain` (core/domain/schema.py:148)

**Problem Types:**
- `BinaryProblem` - Questions with Yes/No options and optional ground truth (core/domain/schema.py:21)
- `OpenEndedProblem` - Questions without predefined answers (core/domain/schema.py:116)

Both problem types have a `to_sample()` method to convert them to `Sample` objects for inference.

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

Policies are unified interfaces for language models. Base class: `Policy` (core/policy/schema.py:49)

**Available Implementations:**
- **`apimodel.py`** - Standard API-based models (OpenAI, Anthropic, DeepSeek, etc.)
- **`raymodel.py`** - Ray-parallelized API calls for high-throughput workloads (>100k tokens/s)
- **`batchmodel.py`** - Provider batch APIs for 50% cost reduction (24-48hr latency)
- **`localmodel.py`** - Local deployment with SGLang backend, supports logprobs and training
- **`human.py`** - CLI-based human-in-the-loop policy
- **`claudecode.py`** - Claude Code agent integration

**Primary Inference Methods (Recommended):**
- **`infer(input)`** / **`infer_async(input)`** - Flexible single inference
  - Accepts: `str | list[dict] | Sample | ProblemDomain`
  - Returns: `str` (for history) or `SingleSample` (for Sample/ProblemDomain)

- **`infer_many(input)`** / **`infer_many_async(input)`** - Flexible batch inference
  - Accepts: `list[str] | list[list[dict]] | list[Sample] | tuple[ProblemDomain, int]`
  - Returns: `list[str]` or `list[SingleSample]`

**Specialized Inference Methods (For Simple History → String):**
- `infer_from_history(history)` / `infer_from_history_async(history)` - Single history → string
- `infer_from_histories(histories)` / `infer_from_histories_async(histories)` - Multiple histories → strings

**Other Key Methods:**
- `logprobs_single(dialogue)` / `logprobs_batch(dialogues)` - Get log probabilities (local models only)
- `train_sft(samples)` / `train_rl(samples, grader)` - Train the model (out-of-place, returns new policy)
- `add_few_shot_examples(examples)` - Create policy with few-shot context (out-of-place)
- `embed(texts)` - Generate embeddings (where supported)

**Sample Types** (core/policy/schema.py:21-47):
- `Sample` - Abstract base with history
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

Trainers orchestrate the training process. Base class: `Trainer` (core/trainer/schema.py:61)

**Available Implementations:**
- **`sft.py`** - Supervised fine-tuning on samples
  - Accepts `list[SingleSample]` directly (no selection/filtering)
  - Supports OpenAI/Together APIs and local training (TRL + DeepSpeed)
  - Automatic validation set creation (none/train/gt strategies)
  - WandB logging support

- **`rl.py`** - Reinforcement learning with custom graders
  - Accepts problem lists and grader
  - Supports OpenAI RL API and local training (TRL GRPO)
  - Works with any `Grader` implementation
  - Configurable KL penalty and reward shaping

- **`fewshot.py`** - Few-shot in-context learning
  - Accepts `list[SingleSample]` directly (no selection/filtering)
  - Creates new policy with prepended context (out-of-place)

**Key Methods:**
- `train(policy, samples, **kwargs)` - Main training entry point (for SFT/FewShot)
- `train(policy, problem_list, grader, **kwargs)` - Main training entry point (for RL)

**Configuration** (core/trainer/schema.py:23):
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

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `TOGETHER_API_KEY` - Together AI API key
- `DEEPSEEK_API_KEY` - DeepSeek API key
- `GOOGLE_API_KEY` - Google (Gemini) API key
- `HUGGINGFACE_API_KEY` - HuggingFace API key
- `OPENROUTER_API_KEY` - OpenRouter API key
- `WANDB_API_KEY` - Weights & Biases logging (optional)

### Training Configuration

- `VALIDATION_STRATEGY` - Validation set strategy: "none", "train", "gt" (default: "none")
- `LORA_RANK` - LoRA rank for parameter-efficient training (default: 0, full-parameter)
- `TRAINED_POLICY_NAME_PATTERN` - Naming pattern for trained models (supports placeholders)
- `GRADER_TYPE` - Grader type: "python_brier", "model_brier", "model_agreement", "model"
- `GRADER_MODEL` - Model name for model-based graders (default: "o4-mini")
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

policy = BatchModel("o4-mini")
# Same interface as other policies, but uses batch API
```

### Ray-based Parallelization

For high-throughput workloads (>100k tokens/s):

```python
from core.policy.raymodel import RayModel

policy = RayModel("o4-mini")
# Automatically parallelizes API calls across workers
```

## License

MIT

## Acknowledgment

Huge thank-you to developers of [safety-research/safety-tooling](https://github.com/safety-research/safety-tooling), which this project is partially based on. 