"""
Model-based agreement grader that uses an LLM to assess agreement between responses and ground truth.
"""

from typing import Any

from core.grader.model_grader import ModelGrader
from core.policy.schema import SingleSample


class ModelAgreementGrader(ModelGrader):
    """
    Grader that uses a model to assess reasoning coverage between a response and ground truth.

    This grader asks a model to:
    1. Extract all key insights, arguments, and considerations from the reference answer
    2. Check how many of these reasoning elements appear in the provided response
    3. Return a score between 0 and 1 indicating the proportion of reference insights discovered

    Key features:
    - Asymmetric evaluation: only deducts points for missing reference insights
    - Conclusion-agnostic: different conclusions are acceptable if reasoning is discovered
    - Extra insights in the response do not reduce the score
    """

    def __init__(
        self,
        model: str = "o4-mini-2025-04-16",
        sampling_params: dict[str, Any] | None = None,
    ):
        """
        Initialize a model-based agreement grader.

        :param model: Model name to use for grading
        :param sampling_params: Sampling parameters for the model
        """
        # Define the grading template that assesses reasoning coverage
        input_template = [
            {
                "role": "system",
                "content": """You are an expert grader evaluating how comprehensively a response discovers the key reasoning elements present in a reference answer.

Your task is to assess REASONING COVERAGE, not conclusion agreement.

CRITICAL INSTRUCTIONS:
1. Extract ALL key insights, arguments, and considerations from the REFERENCE answer
2. Check how many of these reasoning elements appear in the PROVIDED response
3. This is an ASYMMETRIC evaluation:
   - Deduct points ONLY when the provided response misses insights from the reference
   - DO NOT deduct points if the provided response includes additional insights not in the reference
   - DO NOT deduct points if the provided response reaches a different conclusion
4. Focus ONLY on the discovery of reasoning elements, NOT on whether conclusions match

Scoring:
- 1.0: Discovers ALL key insights/arguments/considerations from the reference
- 0.8-0.9: Discovers MOST key reasoning elements (missing 1-2 minor points)
- 0.6-0.7: Discovers MANY key reasoning elements (missing several important points)
- 0.4-0.5: Discovers SOME key reasoning elements (missing about half)
- 0.2-0.3: Discovers FEW key reasoning elements (missing most)
- 0.0-0.1: Discovers NONE or almost none of the key reasoning elements

Output: A single decimal number between 0 and 1""",
            },
            {
                "role": "user",
                "content": """Problem: {{ item.messages[0].content if item.messages else item.question }}

PROVIDED RESPONSE (to be evaluated): {{ sample.output_text }}

REFERENCE ANSWER (source of key insights): {{ item.ground_truth }}

EVALUATION PROCESS:
Step 1: List ALL key insights, arguments, and considerations from the REFERENCE answer
Step 2: For each key element, check if it appears in the PROVIDED response
Step 3: Calculate the proportion of reference insights discovered

IMPORTANT REMINDERS:
- You are evaluating if the PROVIDED response discovered the reasoning from the REFERENCE
- This is NOT bidirectional - extra insights in the PROVIDED response are fine
- Different conclusions are acceptable if the key reasoning was discovered
- Focus on substantive reasoning elements, not stylistic choices

Reasoning Coverage Score (0 to 1):""",
            },
        ]

        # Initialize parent ModelGrader
        super().__init__(
            model=model,
            input_template=input_template,
            name=f"agreement_{model}_grader",
            score_range=(0, 1),  # Agreement scores from 0 to 1
            sampling_params=sampling_params or {"temperature": 0.0, "max_tokens": 10},
        )

    def to_openai_spec(self) -> dict[str, Any]:
        """
        Convert to OpenAI's grader specification format.

        :return: Dictionary conforming to OpenAI's grader API specification
        """
        spec = super().to_openai_spec()
        # Override the name to indicate this is an agreement grader
        spec["name"] = f"model_agreement_{self.model}"
        # Remove sampling_params as it's not supported in OpenAI's grader spec
        if "sampling_params" in spec:
            del spec["sampling_params"]
        return spec

    def validate_problem(self, problem: Any) -> bool:
        """
        Validate that the problem has ground_truth in aux_info.

        :param problem: The problem to validate
        :return: Whether the problem passes validation
        """
        # Check if it has a question
        if not hasattr(problem, "question"):
            return False

        # Check if it has aux_info
        if not hasattr(problem, "aux_info") or problem.aux_info is None:
            return False

        # Check if aux_info has ground_truth
        has_ground_truth = "ground_truth" in problem.aux_info

        # At least one of ground_truth should be present
        return has_ground_truth

    def transform_dataset(self, problems: list) -> list:
        """
        Transform problems to ensure aux_info contains the necessary fields for agreement grading.

        :param problems: List of Problem objects
        :return: List of transformed Problem objects
        """
        from copy import deepcopy

        transformed = []
        for problem in problems:
            # Make a deep copy to avoid modifying the original
            problem_copy = deepcopy(problem)

            # If the problem doesn't have aux_info, initialize it
            if not hasattr(problem_copy, "aux_info") or problem_copy.aux_info is None:
                problem_copy.aux_info = {}

            transformed.append(problem_copy)

        return transformed

    async def grade_async(self, sample: SingleSample, item: dict[str, Any] | None = None) -> float:
        """
        Grade a sample by assessing reasoning coverage compared to ground truth.

        Evaluates how comprehensively the response discovers key insights, arguments,
        and considerations from the reference answer. This is an asymmetric evaluation
        that only deducts points for missing reference insights, not for additional ones.

        :param sample: The single sample to grade (containing history and output)
        :param item: Optional additional information including ground truth
        :return: Reasoning coverage score between 0 and 1
        """
        # Ensure item contains some reference for agreement comparison
        if item is None:
            item = {}

        # Add fields from sample's aux_info if available
        if hasattr(sample, "aux_info") and sample.aux_info:
            if "ground_truth" in sample.aux_info and "ground_truth" not in item:
                item["ground_truth"] = sample.aux_info["ground_truth"]

        # Check if we have at least one reference point
        has_ground_truth = "ground_truth" in item

        if not has_ground_truth:
            raise ValueError("ModelAgreementGrader requires 'ground_truth' in item or sample.aux_info")

        # Add question from history if available
        if "question" not in item and sample.history:
            # Find the last user message as the question
            for msg in reversed(sample.history):
                if msg.get("role") == "user":
                    item["question"] = msg.get("content", "")
                    break

        # Use parent's grade method which handles the model call
        return await super().grade_async(sample, item)
