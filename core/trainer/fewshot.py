"""
Few-shot trainer.
Takes top-scoring trajectories and formats them as in-context examples.
"""

import dataclasses
import os
import random
from typing import Optional

from core.policy.schema import Policy
from core.reasoning.schema import ReasoningMode, ReasoningTrajectory
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
        return len(parts) == 3 and parts[0] == "fewshot" and parts[1].startswith("n") and parts[2].startswith("p")

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
        trajectory_score_files: list[str],
        reasoning_mode: Optional[ReasoningMode] = None,
        **kwargs,
    ) -> Policy:
        """
        Create a new policy with few-shot examples from top trajectories.

        :param policy: The base policy to add few-shot examples to
        :param trajectory_score_files: Paths to the sorted trajectories files
        :param reasoning_mode: The reasoning mode used (for formatting examples)
        :param kwargs: Additional arguments
        :return: New policy with few-shot examples
        """
        # Check if validation strategy is set and warn if not 'none'
        if self.config.validation_strategy != "none":
            logger.urgent(
                f"WARNING: FewShotTrainer ignoring validation_strategy='{self.config.validation_strategy}'. Ignoring validation."
            )
        # Select top trajectories. Scores from different files may be incomparable, so we need to select top trajectories from each file separately.
        top_trajectories = []
        total_trajectories = sum(len(self.load_trajectory_scores(score_file)) for score_file in trajectory_score_files)
        for score_file in trajectory_score_files:
            trajectory_score_pairs = self.load_trajectory_scores(score_file)
            if trajectory_score_pairs:
                top_trajectories.extend(
                    self.select_top_trajectories(
                        trajectory_score_pairs,
                        top_percentage=self.config.top_percentage,  # rounded down
                        top_count=int(
                            self.config.top_count * len(trajectory_score_pairs) / total_trajectories + 0.99
                        ),  # rounded up
                        use_min=True,
                    )
                )

        logger.major(
            f"Selected {len(top_trajectories)} top trajectories out of {total_trajectories} total from {len(trajectory_score_files)} files"
        )

        if not top_trajectories:
            raise ValueError(f"Insufficiently many trajectories found in {trajectory_score_files}")

        # Convert trajectories to few-shot examples
        few_shot_examples = self.trajectories_to_few_shot(top_trajectories, reasoning_mode)

        logger.major(f"Created {len(few_shot_examples)} few-shot dialogue turns")

        # Prepare metadata
        metadata = self.build_metadata(
            policy, trajectory_score_files, num_samples=len(few_shot_examples) // 2, **kwargs
        )

        # Add few-shot examples to policy (creates new policy)
        # Note: add_few_shot_examples is synchronous as it doesn't involve training
        logger.minor(f"Adding {len(few_shot_examples)} dialogue turns as few-shot examples...")
        trained_policy = policy.add_few_shot_examples(few_shot_examples, metadata=metadata)

        logger.major(f"Few-shot training completed, trained policy: {trained_policy}")

        return trained_policy

    def trajectories_to_few_shot(
        self, trajectories: list[ReasoningTrajectory], reasoning_mode: Optional[ReasoningMode] = None
    ) -> list[dict[str, str]]:
        """
        Convert trajectories to few-shot dialogue examples.

        For each trajectory, we use reasoning_mode.trajectory_to_samples to get samples,
        then randomly select one sample and format it as a complete dialogue.

        :param trajectories: List of reasoning trajectories
        :param reasoning_mode: The reasoning mode for formatting (required)
        :return: List of dialogue turns in OpenAI format
        """
        if reasoning_mode is None:
            raise ValueError("reasoning_mode is required for few-shot formatting")

        few_shot_examples = []

        for traj in trajectories:
            try:
                # Get all possible samples from this trajectory
                samples = reasoning_mode.trajectory_to_samples(traj)

                if not samples:
                    logger.major("WARNING: No samples generated from trajectory", dedup="message_stem", max_count=5)
                    continue

                # Randomly select one sample to avoid partial duplicates
                selected_sample = random.choice(samples)

                # Convert the sample to a complete dialogue (history + output)
                dialogue = selected_sample.history.copy()
                dialogue.append({"role": "assistant", "content": selected_sample.output})

                # Add all turns to few-shot examples
                few_shot_examples.extend(dialogue)

            except Exception as e:
                import traceback

                logger.major(
                    "WARNING: Failed to format trajectory: {}",
                    str(e) + traceback.format_exc(),
                    dedup="message_stem",
                    max_count=5,
                )
                continue

        return few_shot_examples
