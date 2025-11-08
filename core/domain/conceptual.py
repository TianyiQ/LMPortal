import os
import random
from copy import deepcopy

from core.domain.schema import OpenEndedProblem, Problem, ProblemDomain
from utils.io_utils import load_file


class Conceptual(ProblemDomain):
    """Conceptual reasoning as a problem domain with open-ended questions and no ground truth."""

    def __init__(
        self,
        dataset_file: str = "conceptual.json",
        train_size: float = 0.5,
    ):
        """Instantiate a conceptual problem set.

        :param dataset_file: dataset filepath relative to `data/questions/`, defaults to "conceptual.json"
        :type dataset_file: str, optional
        :param train_size: the portion of samples to serve as training samples, defaults to 1.0
        :type train_size: float, optional
        """
        super().__init__()

        self.train_size = train_size

        # Access conceptual data
        self.dataset_path = os.path.join("data", "questions", dataset_file)
        self.dataset_content = load_file(self.dataset_path)

        # Parse questions - note the data structure is different from research.py
        # conceptual.json has structure: [{"dimensions": [...], "question": "..."}, ...]
        # We create OpenEndedProblem instances (not BinaryProblem)
        self.questions_all = []
        for i, q in enumerate(self.dataset_content):
            aux_info = deepcopy(q)
            aux_info["domain_name"] = "conceptual"

            self.questions_all.append(
                OpenEndedProblem(
                    id=f"conceptual_{i}",
                    question=q["question"],
                    aux_info=aux_info,
                )
            )

        self.make_questions_splits(self.train_size)

    def preprocess_samples(self, samples: list[OpenEndedProblem]) -> list[OpenEndedProblem]:
        """Preprocess the samples before they are put into the queue. Sample count can change at this stage."""
        # No preprocessing needed for conceptual questions - they are already open-ended
        return samples

    def postprocess_sample(self, sample: OpenEndedProblem) -> OpenEndedProblem:
        """Postprocess a sample after it is sampled. Sample count must not change at this stage."""
        # No postprocessing needed
        return sample
