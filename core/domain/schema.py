"""
This file contains the abstract class for a problem domain.
"""

from __future__ import annotations

import abc
import dataclasses
import hashlib
import os
from collections import deque
from copy import deepcopy
from typing import Any, Literal, Optional, Union

import numpy as np

from utils.io_utils import logger


@dataclasses.dataclass
class BinaryProblem:
    id: str
    question: str
    correct_option: Literal[0, 1] | None
    options: tuple[str, str] = ("Yes", "No")
    aux_info: dict[str, Any] = dataclasses.field(
        default_factory=dict
    )  # Empty by default, can be used to store additional information such as date, topic, etc.

    """
    Converting a collection of problems (like forecasting) into strict Binary problems and eanbles operations such as shuffing and evaluation. 
    
    :param correct_option: ground truth, None if absent. 
    :type correct_option: Literal[0, 1] | None

    :param: aux_info 
    """

    def shuffle_options(self, rng: Optional[np.random.Generator] = None) -> BinaryProblem:
        """Shuffle the options of the problem to avoid the position bias.

        Uses the provided RNG when available for determinism; otherwise falls back to NumPy's global generator.
        """

        problem = deepcopy(self)
        _rng = rng or np.random.default_rng(ProblemDomain._DEFAULT_RNG_SEED)
        if float(_rng.random()) < 0.5:
            problem.options = (problem.options[1], problem.options[0])
            if problem.correct_option is not None:
                problem.correct_option = 1 - problem.correct_option

        return problem

    @classmethod
    def calculate_response_accuracy(
        cls, problems: list[BinaryProblem], responses: list[Literal[0, 1]]
    ) -> tuple[float, tuple[float, float]]:
        """
        Given a collection of problems and corresponding responses, calculate the ground-truth accuracy of the responses (the portion of problems they get right), along with its 95% CI. Higher is better.

        :param problems: The list of problems.
        :type problems: list[BinaryProblem]
        :param responses: The list of answers. Must be of the same length as `problems`.
        :type responses: list[Literal[0, 1]]

        :return: Accuracy in [0,1], and its 95% CI (lower bound, upper bound).
        :rtype: tuple[float, tuple[float, float]]
        """
        accu = sum(p.correct_option == r for p, r in zip(problems, responses, strict=False)) / len(problems)
        se = np.sqrt(accu * (1 - accu) / len(problems))
        return accu, (accu - 1.96 * se, accu + 1.96 * se)

    @classmethod
    def calculate_belief_accuracy_loss(
        cls,
        problems: list[BinaryProblem],
        beliefs: list[float],
        metric: Literal["brier", "cross_entropy"],
    ) -> tuple[float, tuple[float, float]]:
        """
        Calculate the accuracy loss of beliefs for a collection of problems, along with its 95% CI. Lower is better.

        :param problems: The list of problems.
        :type problems: list[BinaryProblem]
        :param beliefs: The list of beliefs. Must be of the same length as `problems`.
        :type beliefs: list[float]
        :param metric: The metric to use for calculating accuracy.
        :type metric: Literal["brier", "cross_entropy"]

        :return: Accuracy loss, and its 95% CI (lower bound, upper bound).
        :rtype: tuple[float, tuple[float, float]]
        """

        def loss(p: BinaryProblem, b: float) -> float:
            assert 0 <= b <= 1

            cr = 1 - p.correct_option

            if metric == "brier":
                return (cr - b) ** 2
            elif metric == "cross_entropy":
                return -cr * np.log(b + 1e-18) - (1 - cr) * np.log(1 - b + 1e-18)

        losses = [
            loss(p, b)
            for p, b in zip(problems, beliefs, strict=False)
            if p.correct_option is not None and b is not None
        ]
        avg_loss = sum(losses) / len(losses)
        stddev = np.std(losses)
        se = stddev / np.sqrt(len(losses))
        return avg_loss, (avg_loss - 1.96 * se, avg_loss + 1.96 * se)


@dataclasses.dataclass
class OpenEndedProblem:
    """Open-ended problem with no binary options or correct answer."""

    id: str
    question: str
    aux_info: dict[str, Any] = dataclasses.field(default_factory=dict)


# Type alias: Problem = BinaryProblem | OpenEndedProblem | MultipleChoiceProblem (upcoming) | ...
Problem = Union[BinaryProblem, OpenEndedProblem]


class ProblemDomain(abc.ABC):
    """A problem domain, with a space of questions, the corresponding belief extraction method, and (optionally) a ground truth verification method."""

    # A dictionary of questions splits, with the keys being the split names and the values being the list of problems in the split
    questions_splits: dict[Literal["train", "test"], list[Problem]]

    # Default per-domain RNG seed. Keeping it constant ensures determinism across sessions.
    _DEFAULT_RNG_SEED: int = 20240821

    def __init__(self):
        """Instantiate a problem domain."""
        # A queue of problems that is to be sampled
        self.sample_queue = {
            "train": deque(),
            "test": deque(),
        }
        # Independent RNG per domain instance for deterministic, interleaving-safe sampling
        self._rng: np.random.Generator = np.random.default_rng(self._DEFAULT_RNG_SEED)

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def _hash_question(question: Problem) -> tuple[str, str]:
        """Platform-independent hashing of a question."""
        return (
            hashlib.sha256(question.question.encode()).hexdigest(),
            hashlib.sha256(question.id.encode()).hexdigest(),
        )

    def make_questions_splits(self, train_size: float):
        """Make the questions splits."""
        self.questions_all.sort(key=self._hash_question)
        train_samples = int(len(self.questions_all) * train_size)
        self.questions_splits = {
            "train": self.questions_all[:train_samples],
            "test": self.questions_all[train_samples:],
        }
        logger.major(f"{self.__class__.__name__} training set size: {len(self.questions_splits['train'])}")
        logger.major(f"{self.__class__.__name__} test set size: {len(self.questions_splits['test'])}")

    def postprocess_sample(self, sample: Problem) -> Problem:
        """Postprocess a sample after it is sampled. Sample count must not change at this stage. Each subclass should implement this method to ensure the sample is in the correct format."""
        return sample

    def preprocess_samples(self, samples: list[Problem]) -> list[Problem]:
        """Preprocess the samples before they are put into the queue. Sample count can change at this stage. Each subclass should implement this method to ensure the sample is in the correct format."""
        return samples

    def sample_problems(self, n: int = 1, split: Optional[Literal["train", "test"]] = None) -> list[Problem]:
        """Sample `n` problems from the problem set. Sampling is without replacement, even across calls.

        :param n: the number of problems to sample, defaults to 1
        :type n: int, optional
        :param split: the split to sample from, defaults to "train"
        :type split: Literal["train", "test"], optional
        :return: a list of `n` problems
        :rtype: list[BinaryProblem]
        """
        split = split or os.getenv("DEFAULT_SPLIT", "train")

        while len(self.sample_queue[split]) < n:
            to_add = deepcopy(self.questions_splits[split])
            # Use per-domain RNG to avoid cross-object interference
            self._rng.shuffle(to_add)
            self.sample_queue[split].extend(self.preprocess_samples(to_add))

        return [self.postprocess_sample(self.sample_queue[split].popleft()) for _ in range(n)]
