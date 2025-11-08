"""
Schema for training strategies.
Defines base classes for different training approaches (SFT, few-shot, etc.)
"""

import abc
import dataclasses
import json
import os
from datetime import datetime
from typing import Literal, Optional

from dacite import Config as DaciteConfig
from dacite import from_dict as dacite_from_dict

from core.domain.schema import BinaryProblem, OpenEndedProblem
from core.policy.schema import Policy, SingleSample
from core.reasoning.schema import ReasoningMode, ReasoningStep, ReasoningTrajectory
from core.schema import Config
from utils.io_utils import compute_hash, load_file, logger


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
        trajectory_score_files: list[str],
        reasoning_mode: Optional[ReasoningMode] = None,
        **kwargs,
    ) -> Policy:
        """
        Train a policy using trajectory-score pairs.

        :param policy: The base policy to train
        :param trajectory_score_files: Paths to the sorted trajectories files from TrajectoryScoreAnalyzer
        :param reasoning_mode: The reasoning mode used to generate trajectories (for formatting)
        :param kwargs: Additional training arguments
        :return: The trained policy
        """
        from utils.async_utils import run_coroutine

        return run_coroutine(self.train_async(policy, trajectory_score_files, reasoning_mode, **kwargs))

    @abc.abstractmethod
    async def train_async(
        self,
        policy: Policy,
        trajectory_score_files: list[str],
        reasoning_mode: Optional[ReasoningMode] = None,
        **kwargs,
    ) -> Policy:
        """
        Asynchronously train a policy using trajectory-score pairs.

        :param policy: The base policy to train
        :param trajectory_score_files: Paths to the sorted trajectories files from TrajectoryScoreAnalyzer
        :param reasoning_mode: The reasoning mode used to generate trajectories (for formatting)
        :param kwargs: Additional training arguments
        :return: The trained policy
        """
        raise NotImplementedError

    def load_trajectory_scores(self, filepath: str) -> list[tuple[ReasoningTrajectory, float]]:
        """
        Load trajectory-score pairs from TrajectoryScoreAnalyzer output or simple format.

        :param filepath: Path to sorted_trajectories.json or simple trajectory-score list
        :return: List of (trajectory, score) tuples
        """
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        trajectory_score_pairs = []

        # Handle different data formats
        if isinstance(data, dict) and "trajectory_score_pairs" in data:
            # TrajectoryScoreAnalyzer format: aggregated_sorted_trajectories.json
            # Example: {"trajectory_score_pairs": [{"trajectory": {...}, "score": 0.8}, ...]}
            items = data["trajectory_score_pairs"]
        elif isinstance(data, list):
            # Simple list format: basic trajectory-score pairs
            # Example: [["User: Q\nAssistant: A", 0.9], ...]
            items = data
        else:
            raise ValueError(f"Unsupported data format in {filepath}")

        for item in items:
            # Handle different item formats
            if isinstance(item, tuple) or isinstance(item, list):
                # Simple (trajectory_str, score) format
                if len(item) == 2:
                    traj_str, score = item
                    # Create a simple ReasoningTrajectory from string
                    question = traj_str.split("\n")[0] if "\n" in traj_str else traj_str
                    traj = ReasoningTrajectory(
                        problem=OpenEndedProblem(question=question, id=f"simple_{compute_hash(question, 7)}"),
                        steps=[ReasoningStep(content=traj_str, trainable=True, belief=None)],
                    )
                    trajectory_score_pairs.append((traj, score))
                else:
                    print(f"WARNING: Invalid item format: {item}")
            elif isinstance(item, dict):
                # Dictionary format with trajectory and score
                traj_dict = item.get("trajectory", item)
                score = item.get("score", None)

                try:
                    # Try to parse as ReasoningTrajectory
                    traj = dacite_from_dict(ReasoningTrajectory, traj_dict, config=DaciteConfig(check_types=False))
                    trajectory_score_pairs.append((traj, score))
                except Exception:
                    # Fall back to string representation
                    if isinstance(traj_dict, str):
                        question = traj_dict.split("\n")[0] if "\n" in traj_dict else traj_dict
                        traj = ReasoningTrajectory(
                            problem=OpenEndedProblem(question=question, id=f"simple_{compute_hash(question, 7)}"),
                            steps=[ReasoningStep(content=traj_dict, trainable=True, belief=None)],
                        )
                        trajectory_score_pairs.append((traj, score))
                    else:
                        print(f"WARNING: Failed to parse trajectory: {traj_dict}")
                        continue

        return trajectory_score_pairs

    def select_top_trajectories(
        self,
        trajectory_score_pairs: list[tuple[ReasoningTrajectory, float]],
        top_percentage: float = None,
        top_count: int = None,
        use_min: bool = True,
    ) -> list[ReasoningTrajectory]:
        """
        Select top trajectories based on scores.

        :param trajectory_score_pairs: List of (trajectory, score) tuples, assumed sorted
        :param top_percentage: Percentage of top trajectories to select
        :param top_count: Maximum number of trajectories to select
        :param use_min: If True, use min(top_count, top_percentage * total)
        :return: List of selected trajectories
        """
        if not trajectory_score_pairs:
            return []

        total = len(trajectory_score_pairs)

        # Determine how many to select
        if use_min and top_percentage is not None and top_count is not None:
            n_select = min(top_count, int(total * top_percentage))
        elif top_percentage is not None:
            n_select = int(total * top_percentage)
        elif top_count is not None:
            n_select = min(top_count, total)
        else:
            n_select = total

        # Select top trajectories (already sorted by score, lower is better)
        selected = [traj for traj, score in trajectory_score_pairs[:n_select]]

        return selected

    def build_metadata(self, policy: Policy, trajectory_score_files: list[str], **kwargs) -> dict:
        """
        Build metadata for training.

        :param policy: The base policy to train
        :param trajectory_score_files: Paths to the sorted trajectories files
        :return: Metadata for training
        """
        # Get characteristics from environment variables, which will be used to name the trained model
        characteristics = (
            kwargs.pop("characteristics", os.getenv("TRAINED_POLICY_NAME_PATTERN", "[DIR_NAME]-N=[NUM_TRAIN_SAMPLES]"))
            .replace(
                "[DIR_NAME]", os.getenv("DIR_NAME", os.getenv("DOMAIN_NAME", "UNK")).strip()
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
            "num_trajectory_score_files": len(trajectory_score_files),
            "trajectory_score_files": trajectory_score_files,
            "config": self.config.to_dict(),
            "characteristics": characteristics,
            **kwargs,
        }

        # Extract source run info from trajectory file if available
        metadata["source_runs"] = []
        for score_file in trajectory_score_files:
            try:
                data = load_file(score_file)
                if "metadata" in data:
                    metadata["source_runs"].append(data["metadata"])
            except Exception as e:
                import traceback

                logger.urgent(
                    "WARNING: Failed to load metadata from {}: {}",
                    score_file,
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
            return min(200, total_samples // 10)  # Very large set: at most 1/10, cap at 200

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

    def filter_ground_truth_validation(self, trajectories: list[ReasoningTrajectory]) -> list[ReasoningTrajectory]:
        """
        Filter trajectories for ground truth validation.

        Only includes trajectories where:
        1. Problem has a correct_option label
        2. Final step belief is non-None and aligned with correct option

        :param trajectories: List of reasoning trajectories
        :return: Filtered list of trajectories suitable for ground truth validation
        """
        filtered = []

        for traj in trajectories:
            # Check if problem has ground truth
            if not hasattr(traj.problem, "correct_option") or traj.problem.correct_option is None:
                continue

            # Check if this is a BinaryProblem
            if not isinstance(traj.problem, BinaryProblem):
                continue

            # Check if trajectory has steps
            if not traj.steps:
                continue

            # Get final step belief
            final_belief = traj.steps[-1].belief
            if final_belief is None:
                continue

            # Check alignment: belief < 0.5 if correct is 1, belief > 0.5 if correct is 0
            correct_option = traj.problem.correct_option
            if correct_option == 1 and final_belief < 0.5:
                filtered.append(traj)
            elif correct_option == 0 and final_belief > 0.5:
                filtered.append(traj)

        return filtered
