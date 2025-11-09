import json
import os
from copy import deepcopy

from datasets import load_dataset
from tqdm import tqdm

from core.domain.schema import BinaryProblem, ProblemDomain
from utils.templates.openreview import review_prompt


class OpenReview(ProblemDomain):
    """Predicting paper acceptance as a problem domain, with access to ground truth."""

    def __init__(
        self,
        train_size: float = 0.5,
        anonymize_authors: bool = False,
        hide_reviews: bool = False,
        hide_references: bool = False,
        max_question_length: int = 128000,
    ):
        """Instantiate an OpenReview problem set. It is strongly preferred to pass the anonymization parameters via environment variables rather than directly in the code.

        :param train_size: the portion of samples to serve as training samples, defaults to 0.8
        :type train_size: float, optional
        :param anonymize_authors: whether to anonymize the authors of the papers, defaults to False
        :type anonymize_authors: bool, optional
        :param hide_reviews: whether to hide the reviews of the papers, defaults to False
        :type hide_reviews: bool, optional
        :param hide_references: whether to hide the references of the papers, defaults to False
        :type hide_references: bool, optional
        """
        super().__init__()
        self.train_size = train_size
        self.anonymize_authors = eval(os.getenv("ANONYMIZE_AUTHORS", str(anonymize_authors)))
        self.hide_reviews = eval(os.getenv("HIDE_REVIEWS", str(hide_reviews)))
        self.hide_references = eval(os.getenv("HIDE_REFERENCES", str(hide_references)))
        self.max_question_length = eval(os.getenv("MAX_QUESTION_LENGTH", str(max_question_length)))

        # Access OpenReview dataset from HuggingFace
        # (data/tmp/openreview_sample_dummy.json is only an example; the actual dataset is much larger)
        if not hasattr(OpenReview, "dataset_content"):
            OpenReview.dataset_content = load_dataset("nhop/OpenReview", split="train")

        self.dataset_content = OpenReview.dataset_content

        # Parse questions
        self.questions_all = [
            BinaryProblem(
                id=f"openreview_{q_id}_{q['openreview_submission_id']}",
                question=review_prompt.format(
                    venue=q["venue"],
                    submission_info=json.dumps(
                        OpenReview.__anonymize_paper(
                            q,
                            anonymize_authors=self.anonymize_authors,
                            hide_reviews=self.hide_reviews,
                            hide_references=self.hide_references,
                        ),
                        ensure_ascii=False,
                    ),
                ),
                options=("ACCEPTED", "REJECTED"),
                correct_option=(0 if q["decision"] else 1),
                aux_info=q | {"domain_name": "openreview"},
            )
            for q_id, q in enumerate(tqdm(self.dataset_content, desc="Loading OpenReview"))
            if ("decision" in q and isinstance(q["decision"], bool) and OpenReview.__is_venue_trusted(q["venue"]))
        ]
        self.questions_all.sort(key=lambda x: x.id)  # Sort for deterministic ordering
        pre_filtering_count = len(self.questions_all)
        self.questions_all = [q for q in self.questions_all if len(q.question) <= self.max_question_length]
        print(f"Removed {pre_filtering_count - len(self.questions_all)} overly long questions.")

        self.make_questions_splits(self.train_size)

    @classmethod
    def __anonymize_paper(
        cls,
        paper: dict,
        anonymize_authors: bool = False,
        hide_reviews: bool = False,
        hide_references: bool = False,
    ) -> dict:
        """Anonymize a paper."""
        paper = deepcopy(paper)
        paper["submission_date"] = paper["publication_date"]

        fileds_to_hide = [
            "paperhash",
            "s2_corpus_id",
            "arxiv_id",
            "publication_date",
            "n_citations",
            "n_influential_citations",
            "decision",
            "decision_text",
            "month_since_publication",
            "avg_citations_per_month",
            "openreview_submission_id",
        ]

        if anonymize_authors:
            fileds_to_hide += ["authors"]

        if hide_reviews:
            fileds_to_hide += ["reviews", "comments"]

        if hide_references:
            fileds_to_hide += ["references"]

        return {k: v for k, v in paper.items() if k not in fileds_to_hide}

    @classmethod
    def __is_venue_trusted(cls, venue: str) -> bool:
        """Check if a venue is trusted. This includes a manually filtered list of CORE A/A* venues having over 50 submissions in the dataset."""
        venue = venue.lower()

        if "workshop" in venue:
            return False

        trusted_venues = [
            "iclr_cc",
            # 'robot-learning_org_CoRL', 'neurips_cc', 'auai_org', 'ICAPS', 'thewebconf', 'acmmm' # NeurIPS, WWW, UAI, ICAPS, CoRL, and MM only make public accepted submissions
        ]

        return any(v.lower() in venue for v in trusted_venues)
