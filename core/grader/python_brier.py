"""
Python-based Brier score grader that extracts beliefs and calculates Brier scores.
"""

import dataclasses
import re
from typing import Any

from core.grader.python_grader import PythonGrader
from core.policy.schema import SingleSample


class PythonBrierGrader(PythonGrader):
    """
    Grader that extracts belief probabilities from responses and calculates Brier scores.

    This grader looks for patterns like "\\finalBeliefProb{0.75}" in the response
    and calculates the Brier score based on the ground truth.
    """

    def __init__(self):
        """Initialize the Python Brier grader."""

        # Define the grade function
        def brier_grade(sample: dict, item: dict) -> float:
            """
            Grade based on Brier score of extracted belief.

            :param sample: The sample dict with 'output_text' field
            :param item: The item dict with problem information
            :return: Negative Brier score (higher is better for rewards)
            """
            output_text = sample.get("output_text", "")

            # Extract belief probability from the response
            # Look for pattern like \finalBeliefProb{0.75}
            pattern = r"(finalBeliefProb|\\boxed)\{([0-9.]+)\}"
            match = re.search(pattern, output_text)

            if not match:
                # If no belief found, return worst possible score
                return -1.0

            try:
                # Extract the belief probability
                belief = float(match.group(2))

                # Clamp to [0, 1]
                belief = max(0.0, min(1.0, belief))

                # Get the correct option from item
                # The item should have either 'correct_option' or 'ground_truth'
                correct_option = None

                if "correct_option" in item:
                    correct_option = item["correct_option"]
                elif "ground_truth" in item:
                    correct_option = item["ground_truth"]
                elif "problem" in item and hasattr(item["problem"], "correct_option"):
                    correct_option = item["problem"].correct_option
                else:
                    # Try to extract from messages if available
                    if "messages" in item:
                        for msg in item["messages"]:
                            content = msg.get("content", "")
                            # Look for patterns indicating the correct answer
                            if "correct answer is 0" in content.lower() or "answer: 0" in content.lower():
                                correct_option = 0
                                break
                            elif "correct answer is 1" in content.lower() or "answer: 1" in content.lower():
                                correct_option = 1
                                break

                if correct_option is None:
                    # If we can't determine the correct option, use a default
                    # or return a neutral score
                    return 0.0

                # Calculate Brier score
                # belief is the probability assigned to option 0
                # cr is the correct probability for option 0
                cr = 1 - correct_option  # Convert to probability for option 0
                brier_score = (cr - belief) ** 2

                # Return negative Brier score (so lower Brier = higher reward)
                # Scale to [-1, 0] range for better RL training
                return -brier_score

            except (ValueError, AttributeError):
                # If belief extraction fails, return worst score
                return -1.0

        # Initialize parent with the grade function
        super().__init__(brier_grade)

        # Store the source code for serialization
        self._source_code = '''def grade(sample, item) -> float:
    """
    Grade based on Brier score of extracted belief.
    
    :param sample: The sample dict with 'output_text' field
    :param item: The item dict with problem information
    :return: Negative Brier score (higher is better for rewards)
    """
    import re
    
    output_text = sample.get("output_text", "")
    
    # Extract belief probability from the response
    # Look for pattern like finalBeliefProb{0.75}
    pattern = r'(finalBeliefProb|\\\\boxed)\\{([0-9.]+)\\}'
    match = re.search(pattern, output_text)
    
    if not match:
        # If no belief found, return worst possible score
        return -1.0
    
    try:
        # Extract the belief probability
        belief = float(match.group(2))
        
        # Clamp to [0, 1]
        belief = max(0.0, min(1.0, belief))
        
        # Get the correct option from item
        correct_option = None
        
        if "correct_option" in item:
            correct_option = item["correct_option"]
        elif "ground_truth" in item:
            correct_option = item["ground_truth"]
        elif "problem" in item and hasattr(item["problem"], "correct_option"):
            correct_option = item["problem"].correct_option
        
        if correct_option is None:
            # If we can't determine the correct option, return neutral
            return 0.0
        
        # Calculate Brier score
        # belief is the probability assigned to option 0
        # cr is the correct probability for option 0
        cr = 1 - correct_option  # Convert to probability for option 0
        brier_score = (cr - belief) ** 2
        
        # Return negative Brier score (so lower Brier = higher reward)
        return -brier_score
        
    except (ValueError, AttributeError):
        # If belief extraction fails, return worst score
        return -1.0
'''

    def to_openai_spec(self) -> dict[str, Any]:
        """
        Convert to OpenAI's grader specification format.

        :return: Dictionary conforming to OpenAI's grader API specification
        """
        return {"type": "python", "source": self._source_code}

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
        if problem.correct_option not in [0, 1, None]:  # Allow None for unknown
            return False
        return True

    async def grade_async(self, sample: SingleSample, item: dict[str, Any] | None = None) -> float:
        """
        Grade a sample by extracting belief and calculating Brier score.
        For local execution during LocalModel training.

        :param sample: The single sample to grade (containing history and output)
        :param item: Optional additional information including ground truth
        :return: Negative Brier score
        """
        # Convert sample to dict format expected by grader function
        sample_dict = {"output_text": sample.output}

        # Prepare item dict with problem information
        item_dict = {**(sample.aux_info or {}), **dataclasses.asdict(sample), **(item or {})}

        # Call the grader function
        return self.grading_function(sample_dict, item_dict)

    def transform_dataset(self, problems: list) -> list:
        """
        Transform problems to include instructions for belief output format.

        :param problems: List of Problem objects
        :return: List of transformed Problem objects
        """
        import copy

        transformed = []

        for problem in problems:
            # Create a deep copy to avoid modifying original
            new_problem = copy.deepcopy(problem)

            # Add instructions to the question about output format
            # Get the actual option content for clear instructions
            option_yes = problem.options[0]
            option_no = problem.options[1]

            format_instruction = (
                f"\n\nIMPORTANT: After your analysis, you MUST include your final belief "
                f"in EXACTLY this format: \\finalBeliefProb{{X}} where X is a number between 0 and 1 "
                f"representing your probability that '{option_yes}' is the correct answer.\n\n"
                f"Example outputs:\n"
                f"- If you believe the answer is '{option_yes}' with 75% chance: \\finalBeliefProb{{0.75}}\n"
                f"- If you believe the answer is '{option_no}' with 80% chance: \\finalBeliefProb{{0.2}}\n"
                f"- If you're completely uncertain between '{option_yes}' and '{option_no}': \\finalBeliefProb{{0.5}}"
            )

            new_problem.question = problem.question + format_instruction
            transformed.append(new_problem)

        return transformed
