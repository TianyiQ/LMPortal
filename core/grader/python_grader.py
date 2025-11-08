"""
Python-based grader that uses custom Python functions to evaluate responses.
"""

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any

from core.grader.schema import Grader
from core.policy.schema import SingleSample
from utils.io_utils import logger


class PythonGrader(Grader):
    """Grader that uses a Python function to evaluate responses."""

    def __init__(self, grading_function: Callable[[dict, dict], float]):
        """
        Initialize a Python grader.

        :param grading_function: A callable that takes (sample, item) and returns float
        """

        def _grading_with_logging(sample: dict, item: dict) -> float:
            result = grading_function(sample, item)
            logger.major(
                "[{}] INFO: Grading({}, {}) = {}",
                self.__class__.__name__,
                sample,
                item,
                result,
                len_trunc=1500,
                per_field_len_trunc=500,
                window_size_secs=600,
                per_window_max_count=1,  # Only log one message every 10 minutes
                dedup="message_stem",
            )
            return result

        self.grading_function = _grading_with_logging
        self._source_code = None

        # Try to get source code for serialization
        try:
            self._source_code = inspect.getsource(_grading_with_logging)
        except (OSError, TypeError):
            # Function might be a lambda or dynamically created
            pass

    @classmethod
    def from_source(cls, source: str) -> "PythonGrader":
        """
        Create a Python grader from source code.

        :param source: Python source code defining a grade function
        :return: PythonGrader instance
        """
        # Execute the source to get the grade function
        namespace = {}
        exec(source, namespace)

        if "grade" not in namespace:
            raise ValueError("Python grader source must define a 'grade' function")

        grader = cls(namespace["grade"])
        grader._source_code = source
        return grader

    def to_openai_spec(self) -> dict[str, Any]:
        """
        Convert to OpenAI's grader specification format.

        :return: Dictionary conforming to OpenAI's grader API specification
        """
        if not self._source_code:
            # Try to generate source code
            func_name = self.grading_function.__name__

            if func_name == "grade":
                # Already has the right name, try to get source
                try:
                    self._source_code = inspect.getsource(self.grading_function)
                except Exception as e:
                    raise ValueError("Cannot serialize grader function to OpenAI spec") from e
            else:
                # Need to wrap it
                try:
                    inner_source = inspect.getsource(self.grading_function)
                    self._source_code = f"""def grade(sample, item) -> float:
    # Wrapper for {func_name}
{inner_source}
    return {func_name}(sample, item)
"""
                except Exception as e:
                    raise ValueError("Cannot serialize grader function to OpenAI spec") from e

        return {"type": "python", "source": self._source_code}

    async def grade_async(self, sample: SingleSample, item: dict[str, Any] | None = None) -> float:
        """
        Grade a sample using the Python function.

        :param sample: The single sample to grade (containing history and output)
        :param item: Optional additional information for grading
        :return: A float score
        """
        # Convert SingleSample to the format expected by OpenAI's grader
        sample_dict = {
            "output_text": sample.output,
            "messages": sample.history + [{"role": "assistant", "content": sample.output}],
        }

        # Add any aux_info to the sample dict
        if hasattr(sample, "aux_info") and sample.aux_info:
            sample_dict.update(sample.aux_info)

        item = {**(sample.aux_info or {}), **dataclasses.asdict(sample), **(item or {})}

        # Call the grade function
        assert not inspect.iscoroutinefunction(
            self.grading_function
        ), "PythonGrader's grading function must be synchronous"
        return self.grading_function(sample_dict, item)
