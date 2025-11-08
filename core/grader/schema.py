"""
Grader schema and base classes for RL training evaluation.
"""

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Union

from core.domain.schema import Problem
from core.policy.schema import SingleSample
from utils.async_utils import run_coroutine
from utils.io_utils import logger


class Grader(ABC):
    """Abstract base class for all graders."""

    @abstractmethod
    def to_openai_spec(self) -> dict[str, Any]:
        """
        Convert the grader to OpenAI's grader specification format.

        :return: Dictionary conforming to OpenAI's grader API specification
        """
        pass

    def grade(self, sample: SingleSample, item: dict[str, Any] | None = None) -> float:
        """
        Grade a sample using the model.

        For local RL training, this creates a local policy instance to grade the sample.
        For OpenAI RL training, grading happens on their servers using to_openai_spec().

        :param sample: The evaluated sample to grade
        :param item: Optional additional information for grading
        :return: A float score
        """
        return run_coroutine(self.grade_async(sample, item))

    @abstractmethod
    async def grade_async(self, sample: SingleSample, item: dict[str, Any] | None = None) -> float:
        """
        Grade a sample and return a reward/score.
        """
        raise NotImplementedError

    def validate_problem(self, problem: Problem) -> bool:
        """
        Validate a problem, checking if it satisfies desiderata for this grader.

        :param problem: The problem to validate
        :return: Whether the problem passes validation
        """
        # Default implementation accepts all problems
        return True

    def transform_dataset(self, problems: list) -> list:
        """
        Transform a dataset of problems before training.
        This can be used to add instructions, format requirements, etc.

        :param problems: List of Problem objects
        :return: List of transformed Problem objects
        """
        # Default implementation returns problems unchanged
        return problems


def create_grader_from_spec(spec: Union[str, dict, Callable[[dict, dict], float]]) -> Grader:
    """
    Create a grader from a specification.

    :param spec: Either a dict with grader specification or a callable
    :return: A Grader instance
    """
    if isinstance(spec, str):
        if "{" not in spec:
            # Handle string as grader class name (e.g., "PythonBrierGrader")
            transformed_spec = re.sub(r"(?<!^)(?=[A-Z])", "_", spec).lower().replace("_grader", "")
            return create_grader_from_spec({"type": transformed_spec})

        else:
            # Try to treat it as a dict spec in JSON format
            import json

            try:
                spec = json.loads(spec)
                return create_grader_from_spec(spec)
            except json.JSONDecodeError as e:
                raise ValueError(f"Unknown grader specification: {spec}") from e

    if callable(spec):
        # Wrap callable in PythonGrader
        from core.grader.python_grader import PythonGrader

        return PythonGrader(spec)

    if not isinstance(spec, dict):
        raise ValueError(f"Invalid grader spec type: {type(spec)}. Must be dict or callable.")

    grader_type = spec.get("type")

    if grader_type == "score_model":
        from core.grader.model_grader import ModelGrader

        return ModelGrader(
            model=spec["model"],
            input_template=spec["input"],
            name=spec.get("name", "grader"),
            score_range=spec.get("range", [0, 1]),
            sampling_params=spec.get("sampling_params", {}),
        )
    elif grader_type == "python":
        from core.grader.python_grader import PythonGrader

        return PythonGrader.from_source(spec["source"])
    elif grader_type == "python_brier":
        from core.grader.python_brier import PythonBrierGrader

        return PythonBrierGrader()
    elif grader_type == "model_brier":
        from core.grader.model_brier import ModelBrierGrader

        return ModelBrierGrader(
            model=spec.get("model", os.getenv("GRADER_MODEL", "o4-mini-2025-04-16")),
            sampling_params=spec.get("sampling_params", {}),
        )
    elif grader_type == "model_agreement":
        from core.grader.model_agreement import ModelAgreementGrader

        return ModelAgreementGrader(
            model=spec.get("model", os.getenv("GRADER_MODEL", "o4-mini-2025-04-16")),
            sampling_params=spec.get("sampling_params", {}),
        )
    else:
        raise ValueError(f"Unknown grader type: {grader_type}")


def create_grader_from_env() -> Grader | None:
    """
    Create a grader from environment variables.

    :return: A Grader instance or None if not configured
    """
    if os.getenv("GRADER_SPEC") is not None:
        return create_grader_from_spec(os.getenv("GRADER_SPEC"))

    grader_type = os.getenv("GRADER_TYPE")

    if grader_type == "model":
        from core.grader.model_grader import ModelGrader

        model = os.getenv("GRADER_MODEL", None)
        if model is None:
            model = "o4-mini-2025-04-16"
            logger.major(
                "WARNING: No GRADER_MODEL found in environment variables. Using default grader model: {}",
                model,
                dedup="message_stem",
            )

        logger.major("INFO: Created ModelGrader with model: {}", model, dedup="message_stem")
        return ModelGrader.create_default(model)

    elif grader_type == "python_brier":
        from core.grader.python_brier import PythonBrierGrader

        logger.major("INFO: Created PythonBrierGrader")
        return PythonBrierGrader()

    elif grader_type == "model_brier":
        from core.grader.model_brier import ModelBrierGrader

        model = os.getenv("GRADER_MODEL", None)
        if model is None:
            model = "o4-mini-2025-04-16"
            logger.major(
                "WARNING: No GRADER_MODEL found in environment variables. Using default grader model: {}",
                model,
                dedup="message_stem",
            )

        logger.major("INFO: Created ModelBrierGrader with model: {}", model, dedup="message_stem")
        return ModelBrierGrader(model=model)

    elif grader_type == "model_agreement":
        from core.grader.model_agreement import ModelAgreementGrader

        model = os.getenv("GRADER_MODEL", None)
        if model is None:
            model = "o4-mini-2025-04-16"
            logger.major(
                "WARNING: No GRADER_MODEL found in environment variables. Using default grader model: {}",
                model,
                dedup="message_stem",
            )

        logger.major("INFO: Created ModelAgreementGrader with model: {}", model, dedup="message_stem")
        return ModelAgreementGrader(model=model)

    else:
        raise ValueError(f"Unknown GRADER_TYPE: {grader_type}")
