"""
Model-based Brier score grader that uses an LLM to extract beliefs and calculate Brier scores.
"""

from typing import Any

from core.grader.model_grader import ModelGrader
from core.policy.schema import SingleSample


class ModelBrierGrader(ModelGrader):
    """
    Grader that uses a model to extract belief probabilities and calculate Brier scores.

    This grader asks a model to:
    1. Read the response and estimate the final belief of the responder
    2. Calculate the Brier score based on the ground truth
    """

    def __init__(
        self,
        model: str = "o4-mini-2025-04-16",
        sampling_params: dict[str, Any] | None = None,
    ):
        """
        Initialize a model-based Brier grader.

        :param model: Model name to use for grading
        :param sampling_params: Sampling parameters for the model
        """
        # Define the grading template that extracts belief and calculates Brier score
        input_template = [
            {
                "role": "system",
                "content": """You are an expert grader evaluating responses based on their expressed beliefs and calculating Brier scores.

Your task:
1. Read the response carefully and identify the responder's final belief probability
2. Calculate the Brier score based on the ground truth
3. Output ONLY a single number: the negative Brier score (between -1 and 0)

Brier Score Formula:
- belief = probability the responder assigns to option 0
- cr = correct probability for option 0 (which is 1 - correct_option)
- Brier = (cr - belief)^2

In other words:
- If the correct answer is option 0: cr = 1, so Brier = (1 - belief)^2
- If the correct answer is option 1: cr = 0, so Brier = belief^2

Output the NEGATIVE Brier score (so lower Brier = higher reward).
Output format: A single decimal number between -1 and 0.""",
            },
            {
                "role": "user",
                "content": """Problem: {{ item.messages[0].content }}

Response: {{ sample.output_text }}

Option 0: {{ item.options[0] }}
Option 1: {{ item.options[1] }}

Correct Answer: Option {{ item.correct_option }}

Instructions:
1. Extract the responder's final belief probability for Option 0 from their response.
   Look for explicit probability statements, confidence levels, or patterns like \\finalBeliefProb{X}.
   Note: \\finalBeliefProb{X} represents the probability for Option 0.
   If no explicit belief is stated, infer from the response content.

2. Calculate the Brier score: (random baseline is 0.5^2=0.25, lower is better)
   - Get the belief probability (0 to 1) for Option 0
   - Calculate cr = 1 - correct_option (the correct probability for Option 0)
   - Brier = (cr - belief)^2

3. Output the NEGATIVE Brier score (a single number between -1 and 0, higher is better).

Example calculations:
- If belief=0.8 for Option 0 and correct is Option 0: cr = 1, Brier = (1-0.8)^2 = 0.04, output: -0.04
- If belief=0.3 for Option 0 and correct is Option 1: cr = 0, Brier = (0-0.3)^2 = 0.09, output: -0.09
- If belief=0.5 for Option 0 and correct is Option 0: cr = 1, Brier = (1-0.5)^2 = 0.25, output: -0.25

Output (single number only):""",
            },
        ]

        # Initialize parent ModelGrader
        super().__init__(
            model=model,
            input_template=input_template,
            name=f"brier_{model}_grader",
            score_range=(-1, 0),  # Negative Brier scores
            sampling_params=sampling_params or {"temperature": 0.0, "max_tokens": 10},
        )

    def to_openai_spec(self) -> dict[str, Any]:
        """
        Convert to OpenAI's grader specification format.

        :return: Dictionary conforming to OpenAI's grader API specification
        """
        spec = super().to_openai_spec()
        # Override the type to indicate this is a Brier score grader
        spec["name"] = f"model_brier_{self.model}"
        # Remove sampling_params as it's not supported in OpenAI's grader spec
        if "sampling_params" in spec:
            del spec["sampling_params"]
        return spec

    def validate_problem(self, problem: Any) -> bool:
        """
        Validate that the problem is a BinaryProblem with required fields.

        :param problem: The problem to validate
        :return: Whether the problem passes validation
        """
        # Check if it's a BinaryProblem (has required fields)
        if not hasattr(problem, "question"):
            return False
        if not hasattr(problem, "correct_option"):
            return False
        if not hasattr(problem, "options"):
            return False
        if len(problem.options) != 2:
            return False
        if problem.correct_option not in [0, 1]:
            return False
        return True

    async def grade_async(self, sample: SingleSample, item: dict[str, Any] | None = None) -> float:
        """
        Grade a sample by extracting belief and calculating Brier score.

        For local RL training, this uses the parent's implementation to call the model.
        For OpenAI RL training, grading happens on their servers using the template.

        :param sample: The single sample to grade (containing history and output)
        :param item: Optional additional information including ground truth
        :return: Negative Brier score between -1 and 0
        """
        # Ensure item contains the required fields for Brier scoring
        if item is None:
            item = {}

        # Add required fields from sample's aux_info if available
        if hasattr(sample, "aux_info") and sample.aux_info:
            if "correct_option" in sample.aux_info and "correct_option" not in item:
                item["correct_option"] = sample.aux_info["correct_option"]
            if "options" in sample.aux_info and "options" not in item:
                item["options"] = sample.aux_info["options"]

        # Validate required fields
        if "correct_option" not in item:
            raise ValueError("ModelBrierGrader requires 'correct_option' in item or sample.aux_info")
        if "options" not in item:
            raise ValueError("ModelBrierGrader requires 'options' in item or sample.aux_info")

        # Use parent's grade method which handles the model call
        return await super().grade_async(sample, item)
