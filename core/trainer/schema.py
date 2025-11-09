"""
Schema for training strategies.
Defines base classes for different training approaches (SFT, few-shot, etc.)
"""

import abc
import dataclasses
import os
from typing import Literal

from core.policy.schema import Policy, SingleSample
from core.schema import Config
from utils.io_utils import load_file, logger


@dataclasses.dataclass
class TrainingConfig(Config):
    """Base configuration for training strategies."""

    # Validation strategy: none, train (split from training set), gt (ground truth filtered)
    validation_strategy: Literal["none", "train", "gt"] = "none"

    # LoRA rank (0 for full-parameter)
    lora_rank: int = 0

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Load configuration from environment variables."""
        config = cls()
        # Load validation strategy from env
        validation_strategy = os.getenv("VALIDATION_STRATEGY", "none")
        if validation_strategy in ["none", "train", "gt"]:
            config.validation_strategy = validation_strategy
        if os.getenv("LORA_RANK"):
            config.lora_rank = int(os.getenv("LORA_RANK"))
        return config

    def identifier(self, **kwargs) -> str:
        """Return a string identifier for the config."""
        return "training"

    @classmethod
    def _is_valid_identifier(cls, identifier: str) -> bool:
        """Check if the identifier is a valid TrainingConfig identifier."""
        return identifier == "training"

    @classmethod
    def _parse_identifier(cls, identifier: str, **kwargs) -> "TrainingConfig":
        """Parse a TrainingConfig from its identifier string."""
        if identifier != "training":
            raise ValueError(f"Invalid TrainingConfig identifier: {identifier}")
        return cls()


class Trainer(abc.ABC):
    """Base class for training strategies."""

    def train(
        self,
        policy: Policy,
        samples: list[SingleSample],
        **kwargs,
    ) -> Policy:
        """
        Train a policy using samples.

        :param policy: The base policy to train
        :param samples: List of SingleSample to train on
        :param kwargs: Additional training arguments
        :return: The trained policy
        """
        from utils.async_utils import run_coroutine

        return run_coroutine(self.train_async(policy, samples, **kwargs))

    @abc.abstractmethod
    async def train_async(
        self,
        policy: Policy,
        samples: list[SingleSample],
        **kwargs,
    ) -> Policy:
        """
        Asynchronously train a policy using samples.

        :param policy: The base policy to train
        :param samples: List of SingleSample to train on
        :param kwargs: Additional training arguments
        :return: The trained policy
        """
        raise NotImplementedError

    def build_metadata(
        self, policy: Policy, source_files: list[str] = None, **kwargs
    ) -> dict:
        """
        Build metadata for training.

        :param policy: The base policy to train
        :param source_files: Optional paths to source files (for legacy compatibility)
        :return: Metadata for training
        """
        source_files = source_files or []

        # Get characteristics from environment variables, which will be used to name the trained model
        characteristics = (
            kwargs.pop(
                "characteristics",
                os.getenv(
                    "TRAINED_POLICY_NAME_PATTERN", "[DIR_NAME]-N=[NUM_TRAIN_SAMPLES]"
                ),
            )
            .replace(
                "[DIR_NAME]",
                os.getenv("DIR_NAME", os.getenv("DOMAIN_NAME", "UNK")).strip(),
            )  # For RL, domain plays the role of DIR_NAME as data source
            .replace("[NUM_TRAIN_SAMPLES]", str(kwargs.pop("num_samples", "UNK")))
            .replace("[NUM_VAL_SAMPLES]", str(kwargs.pop("num_val_samples", "UNK")))
            .replace("/", "+")
            .replace(",", "=")
            .replace("*", "~")
            .replace(" ", "")
        )

        metadata = {
            "training_type": self.__class__.__name__,
            "base_model": policy.colloquial_name,
            "config": self.config.to_dict(),
            "characteristics": characteristics,
            **kwargs,
        }

        # Add source file info if provided (for legacy compatibility)
        if source_files:
            metadata["num_source_files"] = len(source_files)
            metadata["source_files"] = source_files

            # Extract source run info from files if available
            metadata["source_runs"] = []
            for source_file in source_files:
                try:
                    data = load_file(source_file)
                    if "metadata" in data:
                        metadata["source_runs"].append(data["metadata"])
                except Exception as e:
                    import traceback

                    logger.urgent(
                        "WARNING: Failed to load metadata from {}: {}",
                        source_file,
                        str(e) + traceback.format_exc(),
                        dedup="message_stem",
                    )

        return metadata

    def determine_validation_size(self, total_samples: int) -> int:
        """
        Determine optimal validation set size based on training set size.

        Balances:
        - Not reducing training set too much
        - Not making validation too time-consuming
        - Large enough for narrow confidence interval

        :param total_samples: Total number of training samples
        :return: Number of samples to use for validation
        """
        if total_samples <= 10:
            return 0  # Too few samples, no validation
        elif total_samples <= 50:
            return min(5, total_samples // 3)  # Small set: at most 1/3 for validation
        elif total_samples <= 200:
            return min(20, total_samples // 4)  # Medium set: at most 1/4 for validation
        elif total_samples <= 1000:
            return min(100, total_samples // 5)  # Large set: at most 1/5 for validation
        else:
            return min(
                200, total_samples // 10
            )  # Very large set: at most 1/10, cap at 200

    def split_train_validation(
        self, samples: list[SingleSample], validation_size: int
    ) -> tuple[list[SingleSample], list[SingleSample]]:
        """
        Split samples into training and validation sets.

        :param samples: List of all samples
        :param validation_size: Number of samples for validation
        :return: Tuple of (train_samples, val_samples)
        """
        if validation_size <= 0 or validation_size >= len(samples):
            return samples, []

        # Shuffle before splitting to ensure randomness
        import random

        shuffled = samples.copy()
        random.shuffle(shuffled)

        # Split
        val_samples = shuffled[:validation_size]
        train_samples = shuffled[validation_size:]

        return train_samples, val_samples
