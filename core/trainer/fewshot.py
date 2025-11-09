"""
Few-shot trainer.
Takes samples and formats them as in-context examples.
"""

import dataclasses
import os
from typing import Optional

from core.policy.schema import Policy, SingleSample
from core.trainer.schema import Trainer, TrainingConfig
from utils.io_utils import logger


@dataclasses.dataclass
class FewShotConfig(TrainingConfig):
    """Configuration for few-shot training."""

    # Few-shot-specific configurations
    top_count: int = 100  # Max number of examples
    top_percentage: float = 0.1  # Or top 10%

    @classmethod
    def from_env(cls) -> "FewShotConfig":
        """Load few-shot configuration from environment variables."""

        # First load base class configs
        base_config = TrainingConfig.from_env()
        # Create FewShotConfig instance with base configs
        config = cls(validation_strategy=base_config.validation_strategy)

        # Load few-shot configs from env
        if os.getenv("FEWSHOT_TOP_COUNT"):
            config.top_count = int(os.getenv("FEWSHOT_TOP_COUNT"))
        if os.getenv("FEWSHOT_TOP_PERCENTAGE"):
            config.top_percentage = float(os.getenv("FEWSHOT_TOP_PERCENTAGE"))

        return config

    def identifier(self, **kwargs) -> str:
        """Return a string identifier for the few-shot config."""
        return f"fewshot-n{self.top_count}-p{self.top_percentage}"

    @classmethod
    def _is_valid_identifier(cls, identifier: str) -> bool:
        """Check if the identifier is a valid FewShotConfig identifier."""
        parts = identifier.split("-")
        return (
            len(parts) == 3
            and parts[0] == "fewshot"
            and parts[1].startswith("n")
            and parts[2].startswith("p")
        )

    @classmethod
    def _parse_identifier(cls, identifier: str, **kwargs) -> "FewShotConfig":
        """Parse a FewShotConfig from its identifier string."""
        parts = identifier.split("-")
        if len(parts) != 3 or parts[0] != "fewshot":
            raise ValueError(f"Invalid FewShotConfig identifier format: {identifier}")

        # Parse values from identifier
        top_count = int(parts[1][1:])  # Remove 'n' prefix
        top_percentage = float(parts[2][1:])  # Remove 'p' prefix

        # Create config with parsed values
        config = cls()
        config.top_count = top_count
        config.top_percentage = top_percentage
        return config


class FewShotTrainer(Trainer):
    """
    Trainer that creates few-shot examples from top-scoring trajectories.
    """

    def __init__(self, config: Optional[FewShotConfig] = None):
        """Initialize few-shot trainer with configuration."""
        self.config = config or FewShotConfig.from_env()

    async def train_async(
        self,
        policy: Policy,
        samples: list[SingleSample],
        **kwargs,
    ) -> Policy:
        """
        Create a new policy with few-shot examples from provided samples.

        :param policy: The base policy to add few-shot examples to
        :param samples: List of SingleSample to use as few-shot examples
        :param kwargs: Additional arguments
        :return: New policy with few-shot examples
        """
        # Check if validation strategy is set and warn if not 'none'
        if self.config.validation_strategy != "none":
            logger.urgent(
                f"WARNING: FewShotTrainer ignoring validation_strategy='{self.config.validation_strategy}'. Ignoring validation."
            )

        if not samples:
            raise ValueError("No samples provided for few-shot training")

        logger.major(f"Received {len(samples)} samples for few-shot examples")

        # Convert samples to few-shot dialogue examples
        few_shot_examples = self.samples_to_few_shot(samples)

        logger.major(f"Created {len(few_shot_examples)} few-shot dialogue turns")

        # Prepare metadata
        metadata = {
            "training_type": self.__class__.__name__,
            "base_model": policy.colloquial_name,
            "num_samples": len(few_shot_examples) // 2,  # Assuming pairs
            **kwargs,
        }

        # Add few-shot examples to policy (creates new policy)
        # Note: add_few_shot_examples is synchronous as it doesn't involve training
        logger.minor(
            f"Adding {len(few_shot_examples)} dialogue turns as few-shot examples..."
        )
        trained_policy = policy.add_few_shot_examples(
            few_shot_examples, metadata=metadata
        )

        logger.major(f"Few-shot training completed, trained policy: {trained_policy}")

        return trained_policy

    def samples_to_few_shot(self, samples: list[SingleSample]) -> list[dict[str, str]]:
        """
        Convert SingleSample instances to few-shot dialogue examples.

        :param samples: List of SingleSample
        :return: List of dialogue turns in OpenAI format
        """
        few_shot_examples = []

        for sample in samples:
            try:
                # Convert the sample to a complete dialogue (history + output)
                dialogue = sample.history.copy()
                dialogue.append({"role": "assistant", "content": sample.output})

                # Add all turns to few-shot examples
                few_shot_examples.extend(dialogue)

            except Exception as e:
                import traceback

                logger.major(
                    "WARNING: Failed to format sample: {}",
                    str(e) + traceback.format_exc(),
                    dedup="message_stem",
                    max_count=5,
                )
                continue

        return few_shot_examples
