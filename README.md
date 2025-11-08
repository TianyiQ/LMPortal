# Language Model Interface for Inference and Training

## Overview

This repository contains tooling code for inference and training of language models. This includes API-based inference and training, as well as local inference and training, through the same interface.

## Installation and Usage

```bash
uv pip install -e lib/safety_tooling
uv pip install -e .
python -m scripts.fetch_forecast_data # if you'd like to use the forecasting domain
```

This kickstarts the web GUI, from which you may configure and launch evaluation/training runs.

Alternatively, you may manually launch evaluation/training runs through CLI:

```bash
# Evaluation
export ALGO_NAME=WorldInTheLoop
export DOMAIN_NAMES=Research,Forecasting
export REASONING_MODE_NAMES=DirectInference,CoT
export POLICY_LIST_MODE=frontier # evaluate all frontier models
export USE_RAY=1 # use Ray for parallelization
export USE_OPENROUTER=1 # use OpenRouter for model routing
python -m scripts.run_reasoning.py

# Analysis
export DIR_NAME=run-XXXXX  # Use your run directory
python -m scripts.run_analyzers.py

# Training
export SFT_USING_EVAL_RESULTS=1
export RL_USING_DOMAIN=1
export DIR_NAME=run-XXXXX  # For SFT/FEWSHOT: directory with trajectory scores
export DOMAIN_NAMES=Forecasting  # For RL: domains to train on
python -m scripts.run_trainers.py
```

The evaluation script produces `reasoning-trajectories-raw.json` and `bias-eval-results-*.json` in the run directory within the `data/runs` folder. The training script uses trajectory scores from analyzers or domains directly for RL training.

Each run directory is named by `run-{RUN_ID}`. After a run is finished, you may rename the second half of the directory name to something more descriptive, and thereby change run ID. You can use the environment variable `RUN_ID=xxx` to set the run ID for the run; if a directory with the same run ID already exists, data from the previous run will be loaded and analyzed, without executing the reasoning again.

You may use the `DIR_NAME` environment variable to set the location where run directories are stored, relative to the `data/runs` folder.

## Core Components Documentation

### Algorithms (`core/algo/`)
- **`accuracy.py`** - Provides `GroundTruthAccuracy` for evaluating model accuracy against ground truth (previously coupled with MartingaleStrategy, now independent)
- **`mutualpredict.py`** - Provides `MutualPredictStrategy` for measuring mutual predictability between models, as described in Wen et al. (2025)
- **`sycoreason.py`** - Provides `SycophanticReasoning` for measuring sycophancy towards user opinion, including both expressed opinion and IQ change due to user (dis)agreement
- **`qualitative.py`** - Provides `QualitativeJudge` for evaluating free-form text responses, where a judge (arbitrary Policy, can be human) looks for truth-seeking qualities in the response
- **`martingale.py`** - Implements martingale-based strategies for bias evaluation and correction
- **`worldintheloop.py`** - Implements world-in-the-loop evaluation strategies
- **`graderwrapper.py`** - Provides `GraderWrapper` for wrapping any Grader (model-based or Python-based) as a DebiasStrategy for evaluation

#### Algorithm Configuration
All evaluation strategies now expose a typed `AlgoConfig` accessible via `strategy.get_config()` and serialized in bias result files under the `config` field.

Config values are set via environment variables - see the [Environment Variables](#environment-variables) section for more details.

Implemented configs:
- `GroundTruthAccuracyConfig` (metric, mode)
- `MartingaleConfig` (belief_change_type, sample_granularity, regularization_type, informative_switch, informative_coef)
- `MutualPredictConfig` (predictor_choice, predicted_choice, target_question_choice, context_question_choice, k_context, trials_per_sample, scoring, predictee_behavior_examples)
- `QualitativeJudgeConfig` (instruction, red_team_mode, include_self_answer, few_shot_strategy, few_shot_source_path, example_recomputation_rounds, few_shot_bootstrap_rounds, bootstrap_respect_permutations)
- `SycophanticReasoningConfig` (metric, mode)
- `GraderWrapperConfig` (grader_spec, grader_type, grader_model)

### Reasoning Modes (`core/reasoning/`)
- **`direct.py`** - Full response as one single step. Reasoning models have two steps: reasoning and response
- **`cot.py`** - Chain of Thought reasoning implementation
- **`debate.py`** - Self-debate reasoning implementation where models argue different positions
- **`bootstrap.py`** - Bootstrap Interview mode that asks auxiliary questions before the main question to build reasoning capacity

#### Reasoning Configuration
All reasoning modes now support typed `ReasoningConfig` subclasses for configuration:
- `DirectConfig` - Configuration for direct inference mode
- `CoTConfig` - Configuration for Chain of Thought mode
- `DebateConfig` - Configuration for self-debate mode (num_turns)
- `BootstrapInterviewConfig` - Configuration for bootstrap interview mode (num_auxiliary_questions, auxiliary_mode, instruction_types)

### Analysis (`core/analyzer/`)
- **`performance_comparison.py`** - Comparing performance between different policies, and testing the soundness of performance scores
- **`evaluation_relationship.py`** - Correlational and causal relationship between scores from different evaluation algorithms
- **`causal_attribution.py`** - How different features causally contribute to the evaluation score
- **`cross_setup_analysis.py`** - Under the same evaluation algorithm, compare scores that the same policy/trajectory get from different evaluation setups
- **`token_level_evidence.py`** - Token-level analysis, e.g. visualizing how much each token contributes to the final score
- **`result_catalog.py`** - Cataloging all run files to be displayed in the web app

#### Analysis Framework
- Data models:
  - `RunFiles`: shared and per-eval-strategy file references in a run directory
  - `Run`: mirrors a single `runs` catalog entry; holds all co-existing eval strategies' configs
  - `Setup`: a condition grouping of runs with identical `(algo, domain, reasoning_mode, system_prompt)`
- Analyzer interface (scale-aware methods; analyzers implement any subset):
  - `analyze_trajectory(trajectory, *, run, eval_strategies)`
  - `analyze_run(run, *, eval_strategies)`
  - `analyze_setup(setup, *, eval_strategies)`
  - `analyze_across_setups(setups, *, eval_strategies)`
- Orchestration utilities (`utils/analyzer_utils.py`):
  - `discover_run(run_dir|bias_file)` → `Run`
  - `discover_runs_in_batch_dir(batch_dir)` → `Run[]`
  - `group_setups(runs)` → `Setup[]`
  - `collect_eval_strategies_from_runs(runs)` → list of AlgoStrRepr
  - `run_all_analyzers_for_two_batches(dir_a, dir_b, clean_output=True)`
  - `run_analyzers_from_env()` entrypoint used by `scripts/run_analyzers.py`
- Output path policy:
  - `data/analysis/<Analyzer>/<algo>/<domain>/<mode>/<prompt>/<AlgoStrRepr>/<model or ALL>/...`
- Running analyzers:
  - `ANALYZERS=all DIR_NAME=batch-qualitative-expanded python -m scripts.run_analyzers`
  - `ANALYZERS=ResultsCatalogAnalyzer,CausalAttributionAnalyzer,CrossSetupAgreementAnalyzer DIR_NAME=batch-qualitative-expanded/rbr-s-s1-e1-b0-p0,batch-qualitative-expanded/rbr-s-s1-e1-b2-p0 python -m scripts.run_analyzers`

### Domains (`core/domain/`)
- **`research.py`** - Self-curated dataset of research questions, with an easy answer and a hard answer
- **`conceptual.py`** - 31 very thorny conceptual or philosophical questions, meant to test the ability for (1) deconfusion, and (2) think outside the Overton window
- **`forecasting.py`** - Forecasting domain for prediction tasks
- **`changemyview.py`** - ChangeMyView domain for opinion change evaluation
- **`openreview.py`** - OpenReview domain for academic paper evaluation

### Policies (`core/policy/`)
- **`apimodel.py`** - Standard API-based model interface for external LLM services
- **`raymodel.py`** - API-based model accelerated with Ray-based parallelization (higher performance than apimodel.py when throughput is higher than 100,000 token/s)
- **`batchmodel.py`** - API-based model using batch API to save costs. Cost reduced by 50%, but one full run takes up to 48hr. Recommended only when pooling many runs together
- **`localmodel.py`** - Locally deployed model with full logprob access using SGLang backend
- **`human.py`** - Implements the Human policy class, where conversations are shown on the command line and the human user types responses
- **`claudecode.py`** - Claude Code integration for interactive coding assistance

### Graders (`core/grader/`)
- **`schema.py`** - Base `Grader` abstract class and factory functions for creating graders
- **`python_brier.py`** - Extracts `\finalBeliefProb{X}` patterns and calculates Brier scores
- **`model_brier.py`** - Uses LLMs to extract beliefs and calculate Brier scores
- **`python_grader.py`** - Base class for Python-based graders executed on OpenAI servers
- **`model_grader.py`** - Base class for model-based graders using LLMs for evaluation

### Training Pipeline (`core/trainer/`)

The training pipeline supports supervised fine-tuning (SFT), reinforcement learning (RL), and few-shot in-context learning approaches to improve model performance based on trajectory-score pairs from evaluation runs.

#### Training Strategies

- **`sft.py`** - Supervised Fine-Tuning trainer that selects top-scoring trajectories and fine-tunes models on them
  - Supports OpenAI and Together AI fine-tuning APIs for API models
  - Supports local fine-tuning with trl and deepspeed for LocalModel
  - Configurable via `SFT_TOP_PERCENTAGE`, `SFT_NUM_EPOCHS`, `SFT_LEARNING_RATE`, etc.
  - **Validation Support**: Automatic validation set creation with three strategies:
    - `none`: No validation (default)
    - `train`: Split a portion from training set
    - `gt`: Use ground-truth aligned samples only

- **`rl.py`** - Reinforcement Learning trainer using custom graders with OpenAI's RL API
  - Supports both Python and model-based graders for reward calculation
  - Configurable via `RL_NUM_EPOCHS`, `RL_LEARNING_RATE`, etc.
  - Uses graders defined in `core/grader/` for scoring model outputs

- **`fewshot.py`** - Few-shot trainer that formats top trajectories as in-context examples
  - Creates new policies with prepended few-shot examples
  - Randomly selects one sample per trajectory to avoid duplicates
  - Configurable via `FEWSHOT_TOP_COUNT`, `FEWSHOT_TOP_PERCENTAGE`

#### Training Architecture

The training system follows a clean separation of concerns:

1. **Trainers** (`core/trainer/`) - Handle data compilation and selection
   - Load trajectory-score pairs from analyzer outputs
   - Select top trajectories based on configuration
   - Convert trajectories to appropriate training formats
   - **Validation Management**: Automatic validation set creation with intelligent sizing
   - **Ground Truth Filtering**: Filter validation samples to only include correctly aligned beliefs

2. **Policies** (`core/policy/`) - Handle model management and training execution
   - Implement `train_sft()` for supervised fine-tuning with validation support
   - Implement `add_few_shot_examples()` for in-context learning
   - Use `deep_copy()` utility for creating trained policy instances with proper naming and metadata
   - **Real-time Monitoring**: Display training/validation losses during fine-tuning
   - **WandB Integration**: Log comprehensive training metrics including losses, learning rate, gradient norms

3. **Reasoning Modes** (`core/reasoning/`) - Handle trajectory-to-sample conversion
   - Each reasoning mode implements `trajectory_to_samples()` to convert trajectories to training samples
   - Respects the `trainable` flag on reasoning steps to exclude system/non-trainable content

#### Training Workflow

1. **Generate trajectories and scores**: Run evaluation with desired algorithms to produce trajectory-score pairs
2. **Run analyzers**: Use `TrajectoryScoreAnalyzer` to aggregate and sort trajectories by score
3. **Train policies**: The training pipeline automatically detects trajectory score files and trains policies:
   ```bash
   # After running evaluation and analysis
   export TRAINERS=SFTTrainer,FewShotTrainer  # or "all"
   export SFT_TOP_PERCENTAGE=0.1  # Train on top 10%
   export FEWSHOT_TOP_COUNT=100   # Use top 100 examples
   python -m scripts.run_analyzers
   ```

4. **Trained model storage**: Trained models are saved in `data/models/` with unique names:
   - Format: `{base_name}-{training_type}-{hash}`
   - Example: `gpt-4.1-mini-sft-a36b1f2c3f84`
   - Metadata saved alongside including training configuration and source files

#### Configuration

Training behavior is controlled via environment variables and `TrainingConfig`:
- `TRAINERS`: Which trainers to use (SFTTrainer, FewShotTrainer, or "all")
- `VALIDATION_STRATEGY`: Validation set strategy ("none", "train", "gt") (default: "none")
  - `none`: No validation set
  - `train`: Split validation from training set
  - `gt`: Use ground-truth aligned samples only for validation
- `SFT_TOP_PERCENTAGE`: Percentage of top trajectories for SFT (default: 0.1)
- `SFT_NUM_EPOCHS`: Number of training epochs (default: 2)
- `SFT_LEARNING_RATE`: Learning rate for fine-tuning (default: 1e-5)
- `FEWSHOT_TOP_COUNT`: Maximum examples for few-shot (default: 100)
- `FEWSHOT_TOP_PERCENTAGE`: Percentage for few-shot selection (default: 0.1)
- `WANDB_API_KEY`: Optional WandB API key for training metrics logging

The system uses the minimum of count and percentage limits for few-shot to avoid excessive context length.

## Codebase structure

#### `core`

Contains the core logic of the project.

- `core/algo`: Contains the debiasing algorithms (e.g. Martingale, justified flipping)
- `core/analyzer`: Contains the result analyzers (e.g. PerformanceComparisonAnalyzer, EvaluationRelationshipAnalyzer)
- `core/domain`: Contains the problem domains (e.g. forecasting)
- `core/policy`: Contains the policy models (e.g. API model, Local LLMs)
- `core/reasoning`: Contains the reasoning modes (e.g. Self-Debate, CoT)

Each of these components has a `schema` file that describes the components' inputs and outputs, accompanied by a number of subclass files that implement the schema for specific algorithms/domains/policies/reasoning modes.

#### `scripts`

Contains scripts for data fetching, processing, and analysis.

- `scripts/run_reasoning.py`: Contains the script for producing reasoning trajectories given any combination of reasoning modes, domains, policies, evaluation algorithms, and analyzers.
- `scripts/data/*`: Contains the scripts for data fetching and organization.
- `scripts/misc/*`: Contains all other scripts of long-standing value.
- `scripts/legacy/*`: Contains all deprecated scripts that are only kept for backward compatibility.

#### `utils`

Contains utility functions for the project.

- `utils/policy_utils.py`: Contains the utility functions for policy model creation and other policy-related operations.
- `utils/io_utils.py`: Contains the utility functions for input/output operations, including the handling of JSON formatting.
- `utils/async_utils.py`: Contains the utility functions for asynchronous operations.
- `utils/stats_utils.py`: Contains tools for statistical analysis and plotting.
- `utils/nlp_utils.py`: Contains tools for natural language processing.
- `utils/path_utils.py`: Expands the PATH variable to include all levels of the project directory.
- `utils/analyzer_utils.py`: Contains utility functions for calling analyzers.
- `utils/debate_processing_utils.py`: Contains the utility functions for processing the debate data.
- `utils/judge_manipulation_utils.py`: Contains the utility functions for manipulating the judge policy's belief, most useful for SelfDebate.
- `utils/killall.sh`: Contains the script for killing all GPU processes, useful for LocalModel.
- `utils/templates/*`: Contains prompt templates.

## Environment Variables

The codebase uses various environment variables for configuration. These should be set as needed before running experiments:


This document lists all environment variables supported by the ERC evaluation system. Variables are grouped into Features, Execution, and Experimental categories.

### Features

These variables control which features to evaluate. They are typically set from the UI selection.
- **`ALGO_NAMES`**: Evaluation strategy(-ies) to use
  - Options: GroundTruthAccuracy, MartingaleStrategy, WorldInTheLoop, QualitativeJudge, MutualPredictStrategy, GraderWrapper
  - Required: Yes
- **`DOMAIN_NAMES`**: Domain name(s) to evaluate
  - Options: Forecasting, OpenReview, CMV, Research
  - Required: Yes
- **`REASONING_MODE_NAMES`**: Reasoning mode name(s)
  - Options: DirectInference, ChainOfThought, SelfDebate, BootstrapInterview
  - Required: Yes
- **`POLICY_LIST`**: Policies to evaluate (comma-separated)
- **`SYSTEM_PROMPT`**: System prompt to test (comma-separated)
- **`ANALYZERS`**: Result analyzers to run
  - Options: PerformanceComparisonAnalyzer, EvaluationRelationshipAnalyzer, CausalAttributionAnalyzer, CrossSetupAgreementAnalyzer, TokenLevelEvidenceAnalyzer
 
### Execution

These variables control how the evaluation runs are executed.
- **`NUM_TRAJECTORIES`**: Number of trajectories to generate per combination (default: differs by algorithm and domain)
  - Type: Number
- **`DIR_NAME`**: Directory name for output, use "/" to indicate subdirectory
  - Default: Generates timestamp-based name
- **`DEBUG`**: Debug level (0=none, 1=basic, 2=verbose)
  - Options: 0, 1, 2
  - Default: 0
- **`SAVE_TO_FILE`**: Back up untruncated console logs to file
  - Type: Boolean (0 or 1)
  - Default: false
- **`USE_RAY`**: Use Ray for distributed processing of API calls
  - Type: Boolean (0 or 1)
  - Default: true
- **`USE_OPENROUTER`**: Use OpenRouter for model routing (requires USE_RAY=true)
  - Type: Boolean (0 or 1)
  - Default: true
- **`USE_BATCH`**: Use provider-specific batch APIs (requires USE_RAY=false)
  - Type: Boolean (0 or 1)
  - Default: false
- **`PARALLEL_BATCH`**: Use async parallelism across runs
  - Type: Boolean (0 or 1)
  - Default: true
- **`MAX_WORKERS`**: Maximum number of Ray workers
  - Type: Number
- **`RERUN_INCOMPLETE`**: Rerun experiments that contain < NUM_TRAJECTORIES trajectories
  - Type: Boolean (0 or 1)
  - Default: true
- **`RECOMPUTE_RESULTS`**: If to recompute and overwrite existing final result JSON file
  - Type: Boolean (0 or 1)
  - Default: false
- **`RECOMPUTE_TRAJECTORIES`**: When to recompute trajectories JSON file
  - Options: never, missing, always
    - `"never"`: Never recompute trajectories, always use existing ones
    - `"missing"`: Only generate trajectories if they don't exist
    - `"always"`: Always regenerate trajectories (expensive)
  - Default: missing
- **`RECOMPUTE_BELIEFS`**: When to recompute beliefs JSON file
  - Options: never, missing, always
    - `"never"`: Never recompute beliefs, always use existing ones
    - `"missing"`: Only measure beliefs if they don't exist
    - `"always"`: Always remeasure beliefs (recommended for experimenting with different algorithms)
  - Default: missing

#### API Keys
- **`OPENROUTER_API_KEY`**: OpenRouter API key
- **`HUGGINGFACE_API_KEY`**: HuggingFace API key
- **`TOGETHER_API_KEY`**: TogetherAI API key
- **`OPENAI_API_KEY`**: OpenAI API key
- **`ANTHROPIC_API_KEY`**: Anthropic API key
- **`GOOGLE_API_KEY`**: Google API key
- **`WANDB_API_KEY`**: Weights & Biases API key for training metrics logging

#### Debug
- **`SHOW_PROGRESS`**: Show progress bars
  - Type: Boolean (0 or 1)
  - Default: true
 
#### Performance
- **`NO_RETRY`**: Disable retry mechanism for API calls
  - Type: Boolean (0 or 1)
  - Default: false

### Experimental

These variables control experimental features and algorithm-specific behaviors.
- **`JUDGE_POLICY_NAMES`**: Judge policy names (comma-separated)
  - Dynamically set based on selected algorithms
- **`TEMPERATURE`**: Model temperature for non-Ray API models
  - Type: Number
  - Default: 0.25
- **`PRESENCE_PENALTY`**: Presence penalty for non-Ray API models
  - Type: Number
  - Default: 0.0
- **`POLICY_LIST_MODE`**: Canonical policy list, overrides POLICY_LIST
  - Options: frontier, legacy, neurips
    - `"frontier"`: Current default policy list (gpt-4.1, gpt-o3, deepseek-v3, llama-4, claude-sonnet-4, etc.)
    - `"legacy"`: Legacy policy list (subset of frontier models)
    - `"neurips"`: All 21 policies from batch-neurips directory including -confirmatory/-critical variants
- **`FORBIDDEN_MODELS`**: Remove policies whose names contain any of the following (comma-separated)

#### Belief Measurement
- **`DISABLE_SYSTEM_PROMPT_IN_BELIEF_MEASUREMENT`**: Disable system prompts for judge policies during belief measurement
  - Type: Boolean (0 or 1)
  - Default: true
- **`USE_FIXED_JUDGE`**: Use fixed judge for evaluation instead of the evaluated policy itself
  - Type: Boolean (0 or 1)
  - Default: true
- **`OBJECTIVE_BELIEF`**: Judge estimates beliefs from the standpoint of the evaluated policy
  - Type: Boolean (0 or 1)
  - Default: true
- **`USE_PER_TRAJ_BELIEF_MEASURE`**: Use per-trajectory belief measurement instead of per-step
  - Type: Boolean (0 or 1)
  - Default: true
- **`DECOUPLE_TRAJECTORY_BELIEFS`**: Save belief measurement and trajectories in separate files
  - Type: Boolean (0 or 1)
  - Default: true
    - `0`: Legacy mode - trajectories and beliefs stored together in `reasoning-trajectories.json`
    - `1`: Decoupled mode - raw trajectories in `reasoning-trajectories-raw.json`, beliefs in `reasoning-beliefs-{algorithm}.json`

#### World-in-the-Loop

Available when WorldInTheLoop algorithm is selected.
- **`WITL_RECOMPUTE_POLICY`**: Which World-in-the-Loop components to recompute
  - Options: investigation, uplift, upliftblanket, all
  - Default: all
- **`FORECASTER_TEMPLATE`**: Template to use for presenting investigation results to forecaster
  - Options: vanilla, rephrase, toolcall
      - `"vanilla"` - Simple prediction prompt with direct investigation result as assistant response
      - `"rephrase"` - Rephrases investigation results to remove style familiarity, includes tool usage explanations
      - `"toolcall"` - Most natural approach: forecaster retrieves investigation report through formal tool call API, mentioning that Claude Code completed the investigation
  - Default: vanilla
- **`WITL_PER_TOKEN_UPLIFT`**: Use per-token uplift calculation
  - Type: Boolean (0 or 1)
  - Default: true

#### Qualitative Judge

Available when QualitativeJudge algorithm is selected.
- **`QUALITATIVE_JUDGE_USE_FEW_SHOT`**: Use few-shot examples in qualitative judge
  - Type: Boolean (0 or 1)
  - Default: true
- **`QUAL_RED_TEAM_MODE`**: Qualitative judge red team mode
  - Options: none, red, red_blue, red_blue_resolution
  - Default: red_blue_resolution
- **`QUAL_FEW_SHOT_PERMUTE`**: Permute few-shot examples
  - Type: Boolean (0 or 1)
  - Default: true
- **`QUAL_EXAMPLE_RECOMPUTATION_ROUNDS`**: Example recomputation rounds
  - Type: Number
  - Default: 1
- **`QUAL_FEW_SHOT_BOOTSTRAP_ROUNDS`**: Few-shot bootstrap rounds
  - Type: Number
  - Default: 1
- **`QUAL_BOOTSTRAP_RESPECT_PERMUTE`**: Respect permutation in bootstrap
  - Type: Boolean (0 or 1)
  - Default: false
- **`FEW_SHOT_SOURCE_PATH`**: Few-shot source path
  - Default: data/questions/conceptual_human_examples.json
- **`FEW_SHOT_BOOTSTRAP_SOURCE_PATH`**: Few-shot bootstrap source path

#### Bootstrap Interview

Available when BootstrapInterview reasoning mode is selected.
- **`BOOTSTRAP_NUM_AUXILIARY`**: Number of auxiliary questions to ask before the main question
  - Type: Number
  - Default: 3
- **`BOOTSTRAP_MODE`**: Method for selecting auxiliary questions
  - Options: fixed_sequence, iid, stationary_ood, llm_preset, llm_adaptive
    - `"fixed_sequence"`: Use predefined list of truth-seeking questions
    - `"iid"`: Sample from same domain as the main question
    - `"stationary_ood"`: Sample from other specified domains
    - `"llm_preset"`: LLM generates all auxiliary questions at once
    - `"llm_adaptive"`: LLM generates questions adaptively based on conversation
  - Default: iid
- **`BOOTSTRAP_OOD_DOMAINS`**: Domains to sample from for stationary_ood mode (comma-separated)
  - Example: Research,Forecasting
- **`BOOTSTRAP_GENERATOR_POLICY`**: Policy to use for generating auxiliary questions in LLM modes
  - Example: gpt-4.1-mini
- **`BOOTSTRAP_INSTRUCTION_TYPES`**: Instruction types for LLM generation (comma-separated)
  - Options: curriculum, contradiction_seeking, synergistic, socratic
    - `"curriculum"`: Build progressively in complexity
    - `"contradiction_seeking"`: Focus on eliciting contradictions
    - `"synergistic"`: Establish cross-domain connections
    - `"socratic"`: Use Socratic questioning to uncover assumptions

#### GraderWrapper

Available when GraderWrapper algorithm is selected.
- **`GRADER_SPEC`**: Grader specification for GraderWrapper (JSON string or env var name)
  - Type: String
  - Used to instantiate an arbitrary grader via create_grader_from_spec
- **`GRADER_TYPE`**: Type of grader to use (if not using GRADER_SPEC)
  - Options: python_brier, model_brier, model
  - Used as fallback if GRADER_SPEC not provided
- **`GRADER_MODEL`**: Model to use for model-based grading
  - Type: String (e.g., o1-mini, gpt-4)
  - Used when GRADER_TYPE is model_brier or model

#### Mutual Predictability

Available when MutualPredictStrategy algorithm is selected.
- **`MP_PREDICTOR_CHOICE`**: MutualPredictStrategy predictor policy choice
  - Options: evaluated, random_non_evaluated, random_any
  - Default: random_non_evaluated
 - **`MP_PREDICTED_CHOICE`**: MutualPredictStrategy predicted policy choice
  - Options: evaluated, random_non_evaluated, random_any
  - Default: evaluated
 - **`MP_TARGET_QUESTION`**: MutualPredictStrategy target question choice
  - Options: evaluated_question, random_non_evaluated_question, random_any_question
  - Default: evaluated_question
 - **`MP_CONTEXT_QUESTIONS`**: MutualPredictStrategy context question choice
  - Options: evaluated_question, k_random_non_evaluated
  - Default: k_random_non_evaluated
 - **`MP_K_CONTEXT`**: Number of context questions
  - Type: Number
  - Default: 3
 - **`MP_TRIALS_PER_SAMPLE`**: Trials per sample to average
  - Type: Number
  - Default: 3
 - **`MP_SCORING`**: MutualPredictStrategy scoring method
  - Options: uplift, conditional_only, judge_consistency
  - Default: uplift
 - **`MP_PREDICTEE_BEHAVIOR_EXAMPLES`**: Number of predictee behavioral examples
  - Type: Number
  - Default: 0
 
#### Performance Comparison Analyzer

Available when PerformanceComparisonAnalyzer is selected.
- **`REMOVE_N_OUTGROUP_SETUP`**: Number of outgroup data points (with most different setup names) to remove from per-setup performance comparison
  - Type: Number
  - Default: 0
 - **`REMOVE_N_OUTGROUP_ACROSS_SETUPS`**: Number of outgroup data points (with most different setup names) to remove from across-setups performance comparison
  - Type: Number
  - Default: 0
 - **`GROUP_MODE_SETUP`**: How to group data points in the per-setup performance comparison ("none", "qwen", "substr_KEY1=VALUE1_...")
  - Default: none
 - **`GROUP_MODE_ACROSS_SETUPS`**: How to group data points in the across-setups performance comparison ("none", "qwen", "substr_KEY1=VALUE1_...")
  - Default: none
 - **`ASSIGN_X_COORDS_SETUP`**: How to assign x coordinates when plotting per-setup performance comparison
  - Options: none, lexical, lexical_and_group, size, size_and_group, size_group_and_random
  - Default: none
 - **`ASSIGN_X_COORDS_ACROSS_SETUPS`**: How to assign x coordinates when plotting across-setups performance comparison
  - Options: none, lexical, lexical_and_group, size, size_and_group, size_group_and_random
  - Default: none
 - **`ELO_CORR_X_TICKS_SETUP`**: How to assign x ticks when plotting per-setup performance comparison ("names" or list of string labels)
  - Default: none
 - **`ELO_CORR_X_TICKS_ACROSS_SETUPS`**: How to assign x ticks when plotting across-setups performance comparison ("names" or list of string labels)
  - Default: none
 - **`ELO_CORR_X_LABEL_SETUP`**: X-axis label for per-setup performance comparison
  - Default: Setup Index
 - **`ELO_CORR_X_LABEL_ACROSS_SETUPS`**: X-axis label for across-setups performance comparison
 - **`ELO_CORR_Y_COORDS_SETUP`**: How to transform y coordinates in the per-setup performance comparison ("none", "negate", "log", comma-separated list of python math functions to apply)
  - Default: none
 - **`ELO_CORR_Y_COORDS_ACROSS_SETUPS`**: How to transform y coordinates in the across-setups performance comparison ("none", "negate", "log", comma-separated list of python math functions to apply)
  - Default: none
 

### Usage Examples

```bash
# Legacy mode (default) - trajectories and beliefs together
export DECOUPLE_TRAJECTORY_BELIEFS=0
python -m scripts.run_reasoning

# Efficient belief recomputation - preserve expensive text generation, always recompute beliefs
export DECOUPLE_TRAJECTORY_BELIEFS=1
export RECOMPUTE_TRAJECTORIES=never
export RECOMPUTE_BELIEFS=always
python -m scripts.run_reasoning

# Generate raw trajectories once, measure beliefs as needed
export DECOUPLE_TRAJECTORY_BELIEFS=1
export RECOMPUTE_TRAJECTORIES=missing
export RECOMPUTE_BELIEFS=missing
python -m scripts.run_reasoning
```

## Misc Features and Notes

### File Structure
When decoupling is enabled, the following files are used:
- **`reasoning-trajectories-raw.json`** - Raw reasoning content without belief measurements
- **`reasoning-beliefs-{algorithm}.json`** - Belief measurements for specific algorithms
- **`reasoning-trajectories.json`** - Legacy format (maintained for backward compatibility)

The system automatically detects and loads legacy files when decoupling is enabled, ensuring full backward compatibility.

### Partial Results Recomputation

The evaluation framework supports selective recomputation of incomplete or failed evaluations through two new environment variables:

#### RECOMPUTE_RESULTS
When `RECOMPUTE_RESULTS=1`, the system will:
- Load existing `bias-eval-results-[ALGO_NAME].json` files instead of skipping them
- Pass the existing results to the algorithm's `compute_loss_async` method
- Allow algorithms to selectively recompute missing or failed components
- Overwrite the existing results file with updated evaluations

This is particularly useful for:
- Completing interrupted World-in-the-Loop evaluations
- Rerunning failed investigations due to network issues or data access problems
- Adding missing uplift calculations to existing evaluations

#### WITL_RECOMPUTE_POLICY (World-in-the-Loop only)
When used with `RECOMPUTE_RESULTS=1`, this controls what gets recomputed in World-in-the-Loop evaluations:

- **`"all"`** (default): Recompute everything (task sampling, investigation, uplift) for trajectories with any missing fields (or missing entire trajectory)
- **`"investigation"`**: Only recompute investigation results (NOT subsequent uplift calculations) for trajectories with missing investigation results (or missing entire trajectory)
- **`"uplift"`**: Only recompute uplift rewards for trajectories that have investigation results but missing uplift values
- **`"upliftblanket"`**: Recompute uplift rewards for all trajectories that have investigation results (including those with existing uplift values)

#### Usage Examples

```bash
# Recompute all missing components in an existing World-in-the-Loop evaluation
RECOMPUTE_RESULTS=1 WITL_RECOMPUTE_POLICY=all python -m scripts.run_reasoning

# Only fill in missing investigation results (useful after data access issues / investigator agent API issues are resolved)  
RECOMPUTE_RESULTS=1 WITL_RECOMPUTE_POLICY=investigation python -m scripts.run_reasoning

# Only compute missing uplift values (when investigations completed but forecaster failed)
RECOMPUTE_RESULTS=1 WITL_RECOMPUTE_POLICY=uplift python -m scripts.run_reasoning

# Recompute all uplift values (useful when forecaster model or parameters changed)
RECOMPUTE_RESULTS=1 WITL_RECOMPUTE_POLICY=upliftblanket python -m scripts.run_reasoning
```

### Local Model Choice

For evaluation strategies that involve log probabilities, we deploy local models to compute the log probabilities.

Availability of models depend on support by SGLang, which we use for deployment. As of Aug 12 2025, we have tested the following local models.

Top-choice models tested to work:
- [Qwen/Qwen3-30B-A3B-Instruct-2507](https://github.com/QwenLM/Qwen3) (30B, LMArena #23; base available)
- [Qwen/Qwen3-0.6B](https://github.com/QwenLM/Qwen3) (0.6B, LMArena #?)
- deepseek-v3 (685B, LMArena #37; base available; FP8 supported)

Second-choice models tested to work:
- [zai-org/GLM-4.5-Air](https://github.com/zai-org/GLM-4.5) (110B, LMArena #23 with reasoning; base available; FP8 supported)
- mistral-small-3.2-24b-instruct-2503 (24B, LMArena #96; base available)
- Llama-3.2-1B-Instruct (1B, LMArena #196; base available)

Otherwise top-choice models not currently supported:
- [Qwen/Qwen3-235B-A22B-Instruct-2507](https://github.com/QwenLM/Qwen3) (235B, LMArena #5)

Note that for mutual predictability/WITL, small and weaker models may sometimes work better as forecaster/judge.

### Mutual Predictability (7-axis framework)

Key idea: measure conditional predictability via log probabilities with configurable axes. A single evaluated policy’s answers over many questions are scored by how much a predictor policy improves likelihood assignment to a target answer when given configurable context.

- Core class: `core/algo/mutualpredict.py` → `MutualPredictStrategy`
- Config: `MutualPredictConfig` with axes:
  - Predictor (Axis 2): `EVALUATED_POLICY` | `RANDOM_NON_EVALUATED` | `RANDOM_ANY`
  - Predicted (Axis 3): `EVALUATED_POLICY` | `RANDOM_NON_EVALUATED` | `RANDOM_ANY`
  - Target question (Axis 4): `EVALUATED_QUESTION` | `RANDOM_NON_EVALUATED_QUESTION` | `RANDOM_ANY_QUESTION`
  - Context questions (Axis 5): `EVALUATED_QUESTION` | `K_RANDOM_NON_EVALUATED` (k hyperparam)
  - Context responses (Axis 6): the evaluated policy’s answers (fixed)
  - Scoring (Axis 7): `UPLIFT` (Δcond−Δbase), `CONDITIONAL_ONLY`, or `JUDGE_CONSISTENCY`
  - Trials: `trials_per_sample` to average randomness
  - Predictee behavior examples: `predictee_behavior_examples` (default 0) adds a clearly labeled block of the target policy’s past Q/A to condition the predictor; 0 adds nothing.

Defaults reproduce the previous behavior (cross-question context from the evaluated policy; uplift scoring).

Minimal example (legacy-like behavior)

```python
from core.algo.mutualpredict import (
    MutualPredictStrategy, MutualPredictConfig,
    PredictorChoice, PredictedChoice,
    TargetQuestionChoice, ContextQuestionChoice,
    ScoringMethod,
)

config = MutualPredictConfig(
    predictor_choice=PredictorChoice.RANDOM_NON_EVALUATED,
    predicted_choice=PredictedChoice.EVALUATED_POLICY,
    target_question_choice=TargetQuestionChoice.EVALUATED_QUESTION,
    context_question_choice=ContextQuestionChoice.K_RANDOM_NON_EVALUATED,
    k_context=3,
    trials_per_sample=3,
    scoring=ScoringMethod.UPLIFT,
    predictee_behavior_examples=0,  # default
)

algo = MutualPredictStrategy(judge_policies=[judge_model], config=config)
loss, details = await algo.compute_loss_async(samples=reasoning_trajectories)
```

Peer-prediction-like setup (same question, cross-model), with predictee examples

```python
config = MutualPredictConfig(
    predictor_choice=PredictorChoice.RANDOM_ANY,
    predicted_choice=PredictedChoice.RANDOM_ANY,
    target_question_choice=TargetQuestionChoice.EVALUATED_QUESTION,
    context_question_choice=ContextQuestionChoice.EVALUATED_QUESTION,
    trials_per_sample=5,
    scoring=ScoringMethod.UPLIFT,
    predictee_behavior_examples=3,
)

algo = MutualPredictStrategy(judge_policies=predictor_pool, config=config)
loss, details = await algo.compute_loss_async(
    samples=reasoning_trajectories,
    participant_policies=predictor_pool,   # non-evaluated participants
    reasoning_mode=reasoning_mode,         # to generate target answers for non-evaluated policies
    domain=domain,
)
```

Notes
- Predictor and predicted must differ; such trials are skipped automatically.
- Exact duplicate answers between context and target are skipped.
- When random choices are used on any axis, set `trials_per_sample > 1` for stable averages.
- If `predicted_choice != EVALUATED_POLICY` or `predictee_behavior_examples > 0` with non-evaluated predictee, provide `reasoning_mode` and `domain` to generate/cache answers.

### Qualitative Judge (prompt design & interaction pipeline)

Key class: `core/algo/qualitative.py` → `QualitativeJudge`

- **What it does**: Grades reasoning trajectories for truth-seeking quality via a judge policy (default: `gpt-o3`).
- **Design axes (switchable)**:
  - **Instruction/rubric**: Controls criteria text injected into prompts.
  - **Judge self-answer**: Judge first answers the question to calibrate “easy vs hard insights”.
  - **Adversarial depth** (`red_team_mode`): `none` | `red` | `red_blue` | `red_blue_resolution`.
  - **Few-shot strategy**: `disabled` | `static` (fixed order) | `permuted` (all permutations × recomputation rounds).
  - **Example recomputation rounds (M)**: Recompute example components across rounds to reduce variance.
  - **Few-shot source**: Path to initial examples.
  - **Iterative few-shot bootstrapping (T rounds)**: Each round uses current few-shot set as context to grade candidates from `samples` and adds ≥ max(1, old_count) new examples (≈ doubling per round).
  - **Robust parsing**: Extracts JSON from imperfect LLM outputs with balanced-brace search and fallbacks.

Defaults mirror the prior behavior: self-answer enabled, full `red_blue_resolution`, few-shot enabled with permutations, `M=1`, no bootstrapping.

Environment variables:

- **Core**
  - `QUAL_INSTRUCTION`: Override rubric text (default: built-in rubric)
  - `QUAL_INCLUDE_SELF_ANSWER`=1|0 (default 1)
  - `QUAL_RED_TEAM_MODE` in `{none, red, red_blue, red_blue_resolution}` (default `red_blue_resolution`)
- **Few-shot**
  - `QUALITATIVE_JUDGE_USE_FEW_SHOT`=1|0 (default 1)
  - `QUAL_FEW_SHOT_PERMUTE`=1|0 (default 1 → `permuted`; 0 → `static`)
  - `QUAL_FEW_SHOT_SOURCE`=path (default `data/questions/conceptual_human_examples.json`)
  - `QUAL_EXAMPLE_RECOMPUTATION_ROUNDS`=M (default 1; also respects legacy `EXAMPLE_RECOMPUTATION_ROUNDS`)
- **Iterative few-shot bootstrapping**
  - `QUAL_FEW_SHOT_BOOTSTRAP_ROUNDS`=T (default 0)
  - `QUAL_BOOTSTRAP_RESPECT_PERMUTE`=1|0 (default 0)
  - `QUAL_FEW_SHOT_BOOTSTRAP_SOURCE`=path (default None)
    - Expected to be a file containing reasoning trajectories or a directory (directly or indirectly) containing reasoning trajectories. Must be supplied if `QUAL_FEW_SHOT_BOOTSTRAP_ROUNDS > 0`.

Minimal usage (CLI):

```bash
# Default qualitative judging (self-answer + red-blue-resolution + permuted few-shot)
export ALGO_NAME=QualitativeJudge
python -m scripts.run_reasoning
```

Configure adversarial depth and few-shot behavior:

```bash
# Red + Blue + Resolution with 2 recomputation rounds and permuted few-shot
export ALGO_NAME=QualitativeJudge
export QUAL_RED_TEAM_MODE=red_blue_resolution
export QUALITATIVE_JUDGE_USE_FEW_SHOT=1
export QUAL_FEW_SHOT_PERMUTE=1
export QUAL_EXAMPLE_RECOMPUTATION_ROUNDS=2
python -m scripts.run_reasoning
```

Disable few-shot, run red-team only:

```bash
export ALGO_NAME=QualitativeJudge
export QUALITATIVE_JUDGE_USE_FEW_SHOT=0
export QUAL_RED_TEAM_MODE=red
python -m scripts.run_reasoning
```

Iterative few-shot bootstrapping (doubles examples each round using `samples` as candidates):

```bash
export ALGO_NAME=QualitativeJudge
export QUAL_FEW_SHOT_BOOTSTRAP_ROUNDS=2
# Optional: keep permuted context during bootstrapping rounds
export QUAL_BOOTSTRAP_RESPECT_PERMUTE=1
python -m scripts.run_reasoning
```

Python API (advanced):

```python
from core.algo.qualitative import QualitativeJudge

algo = QualitativeJudge(judge_policies=[judge_model])  # env vars control axes
loss, details = await algo.compute_loss_async(samples=reasoning_trajectories)
```

### Multi-GPU Training with DeepSpeed and Accelerate

The `LocalModel` class supports distributed training across multiple GPUs.

The system automatically detects available GPUs and configures training appropriately:
- Single GPU: Standard training
- Multiple GPUs: Distributed Data Parallel (DDP) training
- With DeepSpeed: ZeRO optimization stages 2 or 3

Two DeepSpeed configurations are provided:
- `data/config/deepspeed_zero2.json`: ZeRO Stage 2 (recommended for most cases)
- `data/config/deepspeed_zero3.json`: ZeRO Stage 3 (for very large models)

A series of Accelerate configurations are provided:
- `data/config/accelerate_config_1node_{N}gpu.yaml`: Pre-configured for single-node, N-GPU setup

#### Environment Variables

Control multi-GPU behavior with these environment variables:

```bash
# Force single GPU usage (useful for debugging)
export FORCE_SINGLE_GPU=1

# Disable DeepSpeed (use regular DDP)
export DISABLE_DEEPSPEED=1

# Control concurrent local model instances
export LOCALMODEL_MAX_CONCURRENT=2
```

#### Basic Usage (Automatic Configuration)

The system automatically detects and uses available GPUs:

```python
from utils.policy_utils import create_policy_from_string

# Create model - will auto-detect GPUs
model = create_policy_from_string("meta-llama/Llama-3.2-1B-Instruct")

# Train with SFT - automatically uses all available GPUs
trained_model = await model.train_sft_async(
    samples=training_samples,
    validation_samples=validation_samples,
)
```

#### Using Accelerate Launch

For explicit control over distributed training:

```bash
# Launch with accelerate (uses config file for single-node, single-GPU setup)
TRAINER_TYPE=rl POLICY_LIST_MODE=Qwen/Qwen3-0.6B DOMAIN_NAME=Forecasting GRADER_TYPE=python_brier RL_NUM_SAMPLES=100 VALIDATION_STRATEGY=train accelerate launch --config_file data/config/accelerate_config_1node_1gpu.yaml -m scripts.run_trainers

# Or configure interactively
accelerate config
TRAINER_TYPE=rl POLICY_LIST_MODE=Qwen/Qwen3-0.6B DOMAIN_NAME=Forecasting GRADER_TYPE=python_brier RL_NUM_SAMPLES=100 VALIDATION_STRATEGY=train accelerate launch -m scripts.run_trainers
```

#### SFT Training with Multi-GPU

```python
from core.trainer.sft import SFTTrainer, SFTConfig

# Configure SFT for multi-GPU
config = SFTConfig(
    num_epochs=2,
    learning_rate=2e-5,
    batch_size=4,  # Per-device batch size
    gradient_accumulation_steps=2,
)

trainer = SFTTrainer(config)
trained_policy = await trainer.train_async(
    policy=model,
    trajectory_score_files=["path/to/trajectories.json"],
)
```

#### RL Training with Multi-GPU

```python
from core.trainer.rl import RLTrainer, RLConfig
from core.grader.python_brier import PythonBrierGrader

# Configure RL for multi-GPU
config = RLConfig(
    num_epochs=2,
    learning_rate=1e-6,
    batch_size=2,  # Per-device batch size
    kl_coef=0.1,
)

trainer = RLTrainer(config)
grader = PythonBrierGrader()

trained_policy = await trainer.train_async(
    policy=model,
    problem_list=problems,
    grader=grader,
)
```

#### Technical Details

##### Implementation

The multi-GPU support is implemented in:
- `core/policy/localmodel.py`: `_get_multi_gpu_config()` method
- Automatic batch size calculation based on GPU count
- Dynamic configuration selection based on available resources

##### Integration with TRL

The implementation uses TRL's built-in support for:
- `SFTTrainer`: Supervised fine-tuning
- `GRPOTrainer`: Group Relative Policy Optimization (RL)

Both trainers automatically handle distributed training when configured with DeepSpeed or Accelerate.