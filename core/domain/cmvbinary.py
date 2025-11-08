import json
import os
import random
import re
import time
from typing import Literal

import tqdm

from core.domain.schema import BinaryProblem, Problem, ProblemDomain
from utils.io_utils import load_file


class CMVBinary(ProblemDomain):
    """Change My View (Reddit site for value-laden questions) as a problem domain, with access to "delta(s) from OP" as ground truth."""

    def __init__(
        self,
        dataset_file: str = "changemymind.json",
        train_size: float = 0.5,
    ):
        """Instantiate a CMV problem set.

        :param dataset_file: dataset filepath relative to `data/questions/`, defaults to "changemymind.json"

        :type dataset_file: str, optional
        :param train_size: the portion of samples to serve as training samples, defaults to 0.8
        :type train_size: float, optional
        """
        super().__init__()
        self.train_size = train_size
        # Access cmv data
        # self.dataset_path = os.path.join("data", "questions", dataset_file)
        self.dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../..", "data", "questions", dataset_file)
        )
        self.dataset_content = load_file(self.dataset_path)

        def construct_question(title: str, options: tuple[str, str]) -> str:
            # Remove the complete word "CMV" (or lowercase version) and preceding/trailing punctuation
            title += f" - {options[0].lower()} or {options[1].lower()}?"
            title = re.sub(r"[\.\?\!:,]*[cC][mM][vV][\.\?\!:,]*", "", title)
            title = re.sub(r"\s+", " ", title)
            return title.strip()

        # Parse questions
        self.questions_all = [
            BinaryProblem(
                id=cmv_id,
                question=construct_question(cmv_prob["op-title"], ("Yes", "No")),
                options=("Yes", "No"),
                correct_option=None,
                aux_info={
                    "context": cmv_prob["op-text-body"],
                    "domain_name": "changemyview",
                },
            )
            for cmv_id, cmv_prob in self.dataset_content.items()
            if "op-title" in cmv_prob
        ]

        self.make_questions_splits(self.train_size)

    def postprocess_sample(self, sample: BinaryProblem) -> BinaryProblem:
        """Postprocess a sample after it is sampled. Sample count must not change at this stage."""
        return sample.shuffle_options(self._rng)
