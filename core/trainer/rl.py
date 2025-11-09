"""
Reinforcement Learning (RL) trainer.
Takes trajectories with rewards and trains policies using RL methods.
"""

import dataclasses
import os
from collections.abc import Callable
from typing import Literal, Optional, Union

from core.grader.schema import Grader, create_grader_from_env, create_grader_from_spec
from core.policy.schema import Policy
from core.trainer.schema import Trainer, TrainingConfig
from utils.io_utils import logger


@dataclasses.dataclass
class RLConfig(TrainingConfig):
    """Configuration for reinforcement learning training."""

    # RL-specific configurations
    learning_rate: float = 1e-6  # Lower LR for RL
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    kl_coef: float = 0.1  # KL penalty coefficient
    reward_scale: float = 1.0  # Scale factor for rewards

    # Grader configuration
    grader_type: Literal["model", "python"] = "python"  # Type of grader to use
    grader_model: Optional[str] = None  # Model to use for model grader

    @classmethod
    def from_env(cls) -> "RLConfig":
        """Load RL configuration from environment variables."""
        # First load base class configs
        base_config = TrainingConfig.from_env()
        # Create RLConfig instance with base configs
        config = cls(
            validation_strategy=base_config.validation_strategy,
            lora_rank=base_config.lora_rank,
        )

        # Load RL configs from env
        if os.getenv("RL_LEARNING_RATE"):
            config.learning_rate = float(os.getenv("RL_LEARNING_RATE"))
        if os.getenv("RL_NUM_EPOCHS"):
            config.num_epochs = int(os.getenv("RL_NUM_EPOCHS"))
        if os.getenv("RL_BATCH_SIZE"):
            config.batch_size = int(os.getenv("RL_BATCH_SIZE"))
        if os.getenv("RL_GRADIENT_ACCUMULATION_STEPS"):
            config.gradient_accumulation_steps = int(
                os.getenv("RL_GRADIENT_ACCUMULATION_STEPS")
            )
        if os.getenv("RL_WARMUP_RATIO"):
            config.warmup_ratio = float(os.getenv("RL_WARMUP_RATIO"))
        if os.getenv("RL_KL_COEF"):
            config.kl_coef = float(os.getenv("RL_KL_COEF"))
        if os.getenv("RL_REWARD_SCALE"):
            config.reward_scale = float(os.getenv("RL_REWARD_SCALE"))
        if os.getenv("RL_GRADER_TYPE"):
            config.grader_type = os.getenv("RL_GRADER_TYPE")
        if os.getenv("RL_GRADER_MODEL"):
            config.grader_model = os.getenv("RL_GRADER_MODEL")

        return config

    def identifier(self, **kwargs) -> str:
        """Return a string identifier for the RL config."""
        return f"rl-e{self.num_epochs}-lr{self.learning_rate}-kl{self.kl_coef}"

    @classmethod
    def _is_valid_identifier(cls, identifier: str) -> bool:
        """Check if the identifier is a valid RLConfig identifier."""
        parts = identifier.split("-")
        # Handle scientific notation in learning rate and kl_coef
        return (
            len(parts) >= 5
            and parts[0] == "rl"
            and parts[1].startswith("p")
            and parts[2].startswith("e")
            and parts[3].startswith("lr")
            and parts[4].startswith("kl")
        )

    @classmethod
    def _parse_identifier(cls, identifier: str, **kwargs) -> "RLConfig":
        """Parse an RLConfig from its identifier string."""
        parts = identifier.split("-")

        if not cls._is_valid_identifier(identifier):
            raise ValueError(f"Invalid RLConfig identifier format: {identifier}")

        # Parse values from identifier
        num_epochs = int(parts[2][1:])  # Remove 'e' prefix

        # Handle learning rate which might have scientific notation
        lr_part = parts[3][2:]  # Remove 'lr' prefix
        if len(parts) > 5 and parts[4].replace(".", "").replace("e", "").isdigit():
            lr_part = "-".join([lr_part, parts[4]])
            kl_part_idx = 5
        else:
            kl_part_idx = 4

        learning_rate = float(lr_part)
        kl_coef = float(parts[kl_part_idx][2:])  # Remove 'kl' prefix

        # Create config with parsed values
        config = cls()
        config.num_epochs = num_epochs
        config.learning_rate = learning_rate
        config.kl_coef = kl_coef
        return config


class RLTrainer(Trainer):
    """
    Trainer for reinforcement learning on trajectories with rewards.
    """

    def __init__(self, config: Optional[RLConfig] = None):
        """Initialize RL trainer with configuration."""
        self.config = config or RLConfig.from_env()

    async def train_async(
        self,
        policy: Policy,
        problem_list: list,  # List of Problem objects from domain
        grader: Union[Grader, dict, Callable],
        **kwargs,
    ) -> Policy:
        """
        Train a policy using reinforcement learning with rewards.

        :param policy: The base policy to train
        :param problem_list: List of Problem objects from the domain
        :param grader: A Grader instance, dict with grader spec, or callable for python grader
        :param kwargs: Additional training arguments
        :return: The trained policy
        """
        if not problem_list:
            raise ValueError("No problems provided for RL training")

        logger.major(f"Starting RL training with {len(problem_list)} problems")

        for problem in problem_list:
            assert grader.validate_problem(
                problem
            ), f"Problem {problem} does not satisfy desiderata for grader"

        # Handle validation based on strategy
        validation_samples = None
        training_problems = None

        if self.config.validation_strategy == "none":
            # No validation, use all problems for training
            training_problems = problem_list
            validation_samples = []

        elif self.config.validation_strategy in ["train", "gt"]:
            # Split from training set
            val_size = self.determine_validation_size(len(problem_list))

            if val_size > 0:
                training_problems = problem_list[val_size:]
                validation_samples = problem_list[:val_size]
            else:
                training_problems = problem_list
                validation_samples = []

        else:
            raise ValueError(
                f"Invalid validation strategy: {self.config.validation_strategy}"
            )

        logger.major(
            f"Split {len(problem_list)} problems into {len(training_problems)} training and {len(validation_samples)} validation for RL"
        )

        if grader is None:
            # Try to create from environment
            grader = create_grader_from_env()
            if grader is None:
                raise ValueError(
                    "Grader is required for RL training. Provide either:\n"
                    "1. A Grader instance\n"
                    "2. A dict with grader specification\n"
                    "3. A Python callable for custom grading\n"
                    "4. Set GRADER_TYPE environment variable"
                )
            logger.major(f"Using grader from environment: {os.getenv('GRADER_TYPE')}")

        # Convert to Grader instance if needed
        if not isinstance(grader, Grader):
            grader = create_grader_from_spec(grader)

        grader_type = grader.to_openai_spec().get("type", "unknown")
        logger.major(f"Using grader: {grader_type}")

        # Transform the problems using the grader's transform_dataset method
        # This allows graders like PythonBrierGrader to add format instructions
        logger.major(
            f"Transforming {len(training_problems)} training problems using grader"
        )
        training_problems = grader.transform_dataset(training_problems)

        if validation_samples:
            logger.major(
                f"Transforming {len(validation_samples)} validation problems using grader"
            )
            validation_samples = grader.transform_dataset(validation_samples)

        # Prepare metadata
        metadata = self.build_metadata(
            policy,
            [],
            num_samples=len(training_problems),
            num_val_samples=len(validation_samples) if validation_samples else 0,
            num_epochs=self.config.num_epochs,
            validation_strategy=self.config.validation_strategy,
            lora_rank=self.config.lora_rank,
            grader_type=grader_type,
            **kwargs,
        )

        # Train the policy using RL
        logger.major(
            f"Starting RL training with {len(training_problems)} training problems"
            + (
                f" and {len(validation_samples)} validation problems"
                if validation_samples
                else ""
            )
        )

        trained_policy = await policy.train_rl_async(
            training_problems,
            grader=grader,
            validation_samples=validation_samples,
            metadata=metadata,
        )

        logger.major(f"RL training completed, trained policy: {trained_policy}")
        return trained_policy
