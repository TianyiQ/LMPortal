"""
Supervised Fine-Tuning (SFT) trainer.
Takes samples and fine-tunes a policy on them.
"""

import dataclasses
import os
from typing import Optional

from core.policy.schema import Policy, SingleSample
from core.trainer.schema import Trainer, TrainingConfig
from utils.io_utils import logger


@dataclasses.dataclass
class SFTConfig(TrainingConfig):
    """Configuration for supervised fine-tuning."""

    # SFT-specific configurations
    top_percentage: float = 0.1  # Top 10% by default
    learning_rate: float = 1e-5
    num_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    save_steps: int = 100
    logging_steps: int = 10

    @classmethod
    def from_env(cls) -> "SFTConfig":
        """Load SFT configuration from environment variables."""

        # First load base class configs
        base_config = TrainingConfig.from_env()
        # Create SFTConfig instance with base configs
        config = cls(
            validation_strategy=base_config.validation_strategy,
            lora_rank=base_config.lora_rank,
        )

        # Load SFT configs from env
        if os.getenv("SFT_TOP_PERCENTAGE"):
            config.top_percentage = float(os.getenv("SFT_TOP_PERCENTAGE"))
        if os.getenv("SFT_LEARNING_RATE"):
            config.learning_rate = float(os.getenv("SFT_LEARNING_RATE"))
        if os.getenv("SFT_NUM_EPOCHS"):
            config.num_epochs = int(os.getenv("SFT_NUM_EPOCHS"))
        if os.getenv("SFT_BATCH_SIZE"):
            config.batch_size = int(os.getenv("SFT_BATCH_SIZE"))
        if os.getenv("SFT_GRADIENT_ACCUMULATION_STEPS"):
            config.gradient_accumulation_steps = int(
                os.getenv("SFT_GRADIENT_ACCUMULATION_STEPS")
            )
        if os.getenv("SFT_WARMUP_RATIO"):
            config.warmup_ratio = float(os.getenv("SFT_WARMUP_RATIO"))
        if os.getenv("SFT_WEIGHT_DECAY"):
            config.weight_decay = float(os.getenv("SFT_WEIGHT_DECAY"))
        if os.getenv("SFT_MAX_SEQ_LENGTH"):
            config.max_seq_length = int(os.getenv("SFT_MAX_SEQ_LENGTH"))
        if os.getenv("SFT_SAVE_STEPS"):
            config.save_steps = int(os.getenv("SFT_SAVE_STEPS"))
        if os.getenv("SFT_LOGGING_STEPS"):
            config.logging_steps = int(os.getenv("SFT_LOGGING_STEPS"))

        return config

    def identifier(self, **kwargs) -> str:
        """Return a string identifier for the SFT config."""
        return f"sft-p{self.top_percentage}-e{self.num_epochs}-lr{self.learning_rate}"

    @classmethod
    def _is_valid_identifier(cls, identifier: str) -> bool:
        """Check if the identifier is a valid SFTConfig identifier."""
        parts = identifier.split("-")
        # Handle scientific notation in learning rate (could have extra dash)
        if len(parts) == 5 and parts[4].isdigit():  # e.g., sft-p0.1-e2-lr5e-05
            parts = parts[:3] + ["-".join(parts[3:])]  # Merge lr parts

        return (
            len(parts) == 4
            and parts[0] == "sft"
            and parts[1].startswith("p")
            and parts[2].startswith("e")
            and parts[3].startswith("lr")
        )

    @classmethod
    def _parse_identifier(cls, identifier: str, **kwargs) -> "SFTConfig":
        """Parse an SFTConfig from its identifier string."""
        parts = identifier.split("-")
        # Handle scientific notation in learning rate (could have extra dash)
        if len(parts) == 5 and parts[4].isdigit():  # e.g., sft-p0.1-e2-lr5e-05
            parts = parts[:3] + ["-".join(parts[3:])]  # Merge lr parts

        if len(parts) != 4 or parts[0] != "sft":
            raise ValueError(f"Invalid SFTConfig identifier format: {identifier}")

        # Parse values from identifier
        top_percentage = float(parts[1][1:])  # Remove 'p' prefix
        num_epochs = int(parts[2][1:])  # Remove 'e' prefix
        learning_rate = float(parts[3][2:])  # Remove 'lr' prefix

        # Create config with parsed values
        config = cls()
        config.top_percentage = top_percentage
        config.num_epochs = num_epochs
        config.learning_rate = learning_rate
        return config


class SFTTrainer(Trainer):
    """
    Trainer for supervised fine-tuning on top-scoring trajectories.
    """

    def __init__(self, config: Optional[SFTConfig] = None):
        """Initialize SFT trainer with configuration."""
        self.config = config or SFTConfig.from_env()

    async def train_async(
        self,
        policy: Policy,
        samples: list[SingleSample],
        **kwargs,
    ) -> Policy:
        """
        Fine-tune a policy on provided samples.

        :param policy: The base policy to fine-tune
        :param samples: List of SingleSample to train on
        :param kwargs: Additional training arguments
        :return: The fine-tuned policy
        """
        if not samples:
            raise ValueError("No training samples provided")

        logger.major(f"Received {len(samples)} training samples")

        # Handle validation based on strategy
        validation_samples = None
        training_samples = None

        if self.config.validation_strategy == "none":
            # No validation, use all samples for training
            training_samples = samples
            logger.major(
                f"Using all {len(training_samples)} samples for training (no validation)"
            )

        elif self.config.validation_strategy == "train":
            # Split from training set
            val_size = self.determine_validation_size(len(samples))

            if val_size > 0:
                training_samples, validation_samples = self.split_train_validation(
                    samples, val_size
                )
                logger.major(
                    f"Split {len(samples)} samples into {len(training_samples)} training and {len(validation_samples)} validation"
                )
            else:
                training_samples = samples
                logger.urgent(
                    f"Too few samples ({len(samples)}) for validation split, using all for training"
                )

        elif self.config.validation_strategy == "gt":
            # For GT validation, we would need Problem objects with ground truth
            # Since we only have SingleSamples here, we can't do GT validation
            logger.urgent(
                "WARNING: GT validation strategy not supported when training directly on SingleSamples. "
                "Using all samples for training."
            )
            training_samples = samples

        # Prepare metadata
        metadata = {
            "training_type": self.__class__.__name__,
            "base_model": policy.colloquial_name,
            "num_samples": len(training_samples),
            "num_val_samples": len(validation_samples) if validation_samples else 0,
            "num_epochs": self.config.num_epochs,
            "validation_strategy": self.config.validation_strategy,
            "lora_rank": self.config.lora_rank,
            "config": self.config.to_dict(),
            **kwargs,
        }

        # Fine-tune the policy using async method with validation set
        logger.major(
            f"Starting fine-tuning with {len(training_samples)} training samples"
            + (
                f" and {len(validation_samples)} validation samples"
                if validation_samples
                else ""
            )
        )

        trained_policy = await policy.train_sft_async(
            training_samples, validation_samples=validation_samples, metadata=metadata
        )

        logger.major(f"Fine-tuning completed, trained policy: {trained_policy}")
        return trained_policy
