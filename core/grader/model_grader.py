"""
Model-based grader that uses an LLM to evaluate responses.
"""

import re
from typing import Any

from core.grader.schema import Grader
from core.policy.schema import SingleSample
from utils.io_utils import logger


class ModelGrader(Grader):
    """Grader that uses a model to evaluate responses."""

    def __init__(
        self,
        model: str,
        input_template: list[dict[str, str]],
        name: str = "model_grader",
        score_range: tuple[float, float] = (0, 1),
        sampling_params: dict[str, Any] | None = None,
    ):
        """
        Initialize a model grader.

        :param model: Model name to use for grading
        :param input_template: Template for the grading conversation
        :param name: Name of the grader
        :param score_range: Range of scores the model should output
        :param sampling_params: Sampling parameters for the model
        """
        self.model = model
        self.input_template = input_template
        self.name = name
        self.score_range = score_range
        self.sampling_params = sampling_params or {"temperature": 0.0, "max_tokens": 10}

    @classmethod
    def create_default(cls, model: str = "o4-mini-2025-04-16") -> "ModelGrader":
        """
        Create a default model grader with standard template.

        :param model: Model to use for grading
        :return: ModelGrader instance
        """
        input_template = [
            {
                "role": "system",
                "content": "You are an expert grader evaluating the quality of responses. "
                "Score the response based on accuracy, helpfulness, and alignment with the task. "
                "Output a single number between 0 and 1, where 0 is poor and 1 is excellent.",
            },
            {
                "role": "user",
                "content": "Task: {{ item.messages[-2].content if item.messages else 'Unknown' }}\n\n"
                "Response: {{ sample.output_text }}\n\n"
                "{% if item.reference_answer %}Reference: {{ item.reference_answer }}\n\n{% endif %}"
                "Score (0 to 1):",
            },
        ]

        return cls(
            model=model,
            input_template=input_template,
            name=f"default_{model}_grader",
            score_range=(0, 1),
            sampling_params={"temperature": 0.0, "max_tokens": 10},
        )

    def to_openai_spec(self) -> dict[str, Any]:
        """
        Convert to OpenAI's grader specification format.

        :return: Dictionary conforming to OpenAI's grader API specification
        """
        # OpenAI's grader spec doesn't support sampling_params
        return {
            "type": "score_model",
            "name": self.name,
            "model": self.model,
            "input": self.input_template,
            "range": list(self.score_range),
        }

    async def grade_async(
        self, sample: SingleSample, item: dict[str, Any] | None = None
    ) -> float:
        """
        Grade a sample using the model (async version).

        For local RL training, this creates a local policy instance to grade the sample.
        For OpenAI RL training, grading happens on their servers using to_openai_spec().

        :param sample: The evaluated sample to grade
        :param item: Optional additional information for grading
        :return: A float score
        """
        # Import here to avoid circular dependency
        import dataclasses

        import jinja2

        from utils.policy_utils import create_policy_from_string

        # Create a local policy instance for grading
        grader_policy = create_policy_from_string(self.model)

        # Convert SingleSample to a format similar to OpenAI's grader expectations
        # This mirrors what PythonGrader does
        sample_dict = {
            "output_text": sample.output,
            "messages": sample.history
            + [{"role": "assistant", "content": sample.output}],
        }

        # Add any aux_info to the sample dict
        if hasattr(sample, "aux_info") and sample.aux_info:
            sample_dict.update(sample.aux_info)

        # Build item by merging aux_info, sample fields, and provided item
        # This matches PythonGrader's approach: {**aux_info, **sample_fields, **item}
        merged_item = {}

        # First add aux_info
        if hasattr(sample, "aux_info") and sample.aux_info:
            merged_item.update(sample.aux_info)

        # Then add all sample fields (if it's a dataclass)
        try:
            if dataclasses.is_dataclass(sample):
                merged_item.update(dataclasses.asdict(sample))
            else:
                # If not a dataclass, add available attributes manually
                for attr in ["history", "output", "aux_info"]:
                    if hasattr(sample, attr):
                        merged_item[attr] = getattr(sample, attr)
        except Exception:
            # If we can't convert to dict, just use what we have
            pass

        # Finally override with provided item
        if item:
            merged_item.update(item)

        # Ensure messages is always available in merged_item for templates
        if "messages" not in merged_item:
            merged_item["messages"] = sample.history

        # Prepare the context for template rendering
        # Convert both sample_dict and merged_item to objects with attribute access
        # This allows templates to use dot notation like item.messages[0].content
        context = {
            "sample": type("SampleObj", (object,), sample_dict)(),
            "item": type("ItemObj", (object,), merged_item)(),
        }

        # Render the grading conversation from template
        messages = []
        for msg_template in self.input_template:
            # Create a Jinja2 template from the content
            template = jinja2.Template(msg_template["content"])
            rendered_content = template.render(**context)

            messages.append({"role": msg_template["role"], "content": rendered_content})

        # For local grading, add instruction to use structured output format
        # This ensures consistent parsing and avoids ambiguity
        local_grading_suffix = (
            "\n\nIMPORTANT: You must output your final score in the format \\boxed{score} "
            "where score is a single number. For example: \\boxed{0.75} or \\boxed{0.25}. "
            "Only the number inside \\boxed{} will be used as the score."
        )

        # Add the suffix to the last user message for local grading
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += local_grading_suffix

        # Get grading from the model using async method
        try:
            # Use the async version of infer to avoid event loop conflicts
            response = await grader_policy.infer_from_history_async(
                history=messages,
                temperature=self.sampling_params.get("temperature", 0.0),
            )

            # Parse the score from the response
            score = self._parse_score(response)

            # Ensure score is within the expected range
            min_score, max_score = self.score_range
            if score < min_score or score > max_score:
                logger.urgent(
                    "[{}] ERROR: Score out of range: {} (expected range: {} to {})",
                    self.__class__.__name__,
                    score,
                    min_score,
                    max_score,
                    window_size_secs=600,
                    per_window_max_count=1,
                    dedup="message_stem",
                )
                return None

            logger.major(
                "[{}] INFO: Grading({}, {}) = {}",
                self.__class__.__name__,
                sample,
                item,
                score,
                len_trunc=1500,
                per_field_len_trunc=500,
                window_size_secs=600,
                per_window_max_count=1,  # Only log one message every 10 minutes
                dedup="message_stem",
            )

            return score

        except Exception as e:
            import traceback

            logger.urgent(
                "[{}] ERROR: Failed to grade sample: {}",
                self.__class__.__name__,
                str(e) + traceback.format_exc(),
                window_size_secs=600,
                per_window_max_count=1,
                dedup="message_stem",
            )
            return None

    def _parse_score(self, response: str) -> float:
        """
        Parse a numeric score from the model's response.
        Prioritizes \\boxed{score} format, then falls back to regular number parsing.

        :param response: The model's response string
        :return: Parsed float score
        :raises ValueError: If no valid score found or multiple \\boxed{} patterns
        """
        # First, try to find score in \boxed{} format
        # Use [^}]* instead of [^}]+ to match empty content too
        boxed_pattern = r"\\boxed\{([^}]*)\}"
        boxed_matches = re.findall(boxed_pattern, response)

        if boxed_matches:
            # Check that \boxed{} is unique
            if len(boxed_matches) > 1:
                raise ValueError(
                    f"Multiple \\boxed{{}} patterns found in response. "
                    f"Found {len(boxed_matches)} instances: {boxed_matches}. "
                    f"Response: {response}"
                )

            # Parse the content inside \boxed{}
            boxed_content = boxed_matches[0].strip()

            # Check for empty boxed
            if not boxed_content:
                raise ValueError(
                    f"Empty \\boxed{{}} found. Content must contain a number. "
                    f"Response: {response}"
                )

            try:
                # Try to parse as float
                score = float(boxed_content)
                logger.minor(
                    "[{}] Parsed score from \\boxed{{{}}}: {}",
                    self.__class__.__name__,
                    boxed_content,
                    score,
                    window_size_secs=600,
                    per_window_max_count=1,
                    dedup="message_stem",
                )
                return score
            except ValueError as e:
                raise ValueError(
                    f"Could not parse number from \\boxed{{{boxed_content}}}. "
                    f"Response: {response}"
                ) from e

        # Fallback: Try to find a number in the response (for backward compatibility)
        # This path should rarely be used with the new format
        logger.minor(
            "[{}] No \\boxed{{}} found, falling back to regular number parsing",
            self.__class__.__name__,
            window_size_secs=600,
            per_window_max_count=1,
            dedup="message_stem",
        )

        # Look for decimal numbers
        decimal_pattern = r"-?\d+\.\d+"
        decimal_matches = re.findall(decimal_pattern, response)
        if decimal_matches:
            # Take the last decimal number found (usually the final answer)
            return float(decimal_matches[-1])

        # Look for integers
        integer_pattern = r"-?\d+"
        integer_matches = re.findall(integer_pattern, response)
        if integer_matches:
            # Take the last integer found
            return float(integer_matches[-1])

        # If no number found, raise an error
        raise ValueError(
            f"Could not parse score from response. Expected \\boxed{{score}} format. "
            f"Response: {response}"
        )
