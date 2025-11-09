from typing import Optional

import numpy as np
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from core.domain.schema import OpenEndedProblem, Problem, ProblemDomain
from utils.io_utils import logger


class CMVFreeForm(ProblemDomain):
    """
    Change My View domain using Stanford Human Preferences (SHP) dataset.

    This domain uses the SHP dataset which contains collective human preferences
    over responses to questions/instructions from Reddit's ChangeMyView subreddit.
    Only the top 20% highest score_ratio entries are used to ensure quality.
    """

    # Default domains to use from SHP dataset
    DEFAULT_DOMAINS = ["changemyview"]

    def __init__(
        self,
        domains: Optional[list[str]] = None,
        train_size: float = 0.5,
        score_ratio_percentile: float = 0.8,  # Use top 20% (above 80th percentile)
    ):
        """
        Instantiate a CMVFreeForm problem set using SHP dataset.

        :param domains: List of SHP domains to use (default: ["changemyview"])
        :type domains: Optional[list[str]]
        :param train_size: Portion of samples for training split (default: 0.5)
        :type train_size: float
        :param score_ratio_percentile: Percentile threshold for score_ratio filtering (default: 0.8)
        :type score_ratio_percentile: float
        """
        super().__init__()
        self.domains = domains or self.DEFAULT_DOMAINS
        self.train_size = train_size
        self.score_ratio_percentile = score_ratio_percentile

        # Load and process data using HuggingFace datasets API
        # Cache the dataset at class level to avoid reloading
        if not hasattr(CMVFreeForm, "_cached_datasets"):
            CMVFreeForm._cached_datasets = {}

        self.questions_all = self._load_and_process_data()

        # Create train/test splits
        self.make_questions_splits(self.train_size)

    def _transform_question(self, question: str) -> str:
        """
        Transform a question to a more readable format.
        """
        # Remove moderator footnotes
        question = question.split(" _____ ")[0].split("*This is a footnote")[0]
        return question.strip() + " Am I wrong? Please correct me."

    def _load_and_process_data(self) -> list[Problem]:
        """
        Load and process SHP data from all specified domains.

        :return: List of OpenEndedProblem instances
        """
        all_problems = []

        for domain in self.domains:
            logger.major(f"Loading SHP domain: {domain}")

            # Check if dataset is already cached
            if domain not in CMVFreeForm._cached_datasets:
                try:
                    # Load all splits using HuggingFace datasets API
                    dataset_splits = []
                    for split in ["train", "validation", "test"]:
                        try:
                            ds = load_dataset("stanfordnlp/SHP", data_dir=domain, split=split)
                            dataset_splits.append(ds)
                        except Exception as e:
                            logger.major(f"Could not load {split} split for {domain}: {e}")

                    if not dataset_splits:
                        logger.major(f"No data found for domain {domain}")
                        continue

                    # Combine all splits
                    full_dataset = concatenate_datasets(dataset_splits)
                    CMVFreeForm._cached_datasets[domain] = full_dataset
                    logger.major(f"Loaded and cached {len(full_dataset)} total entries from {domain}")

                except Exception as e:
                    logger.major(f"Failed to load domain {domain}: {e}")
                    logger.major("Please ensure you have the datasets library installed: pip install datasets")
                    continue
            else:
                full_dataset = CMVFreeForm._cached_datasets[domain]
                logger.major(f"Using cached dataset for {domain} with {len(full_dataset)} entries")

            # Convert to list for filtering
            all_entries = list(full_dataset)
            all_entries.sort(key=lambda x: x.get("post_id"))  # Sort for deterministic ordering

            # Filter by score_ratio percentile
            score_ratios = [entry.get("score_ratio", 1.0) for entry in all_entries]
            threshold = np.percentile(score_ratios, self.score_ratio_percentile * 100)

            filtered_entries = [entry for entry in all_entries if entry.get("score_ratio", 1.0) >= threshold]

            logger.major(
                f"Filtered to {len(filtered_entries)} entries with score_ratio >= {threshold:.2f} "
                f"(top {(1 - self.score_ratio_percentile) * 100:.0f}%)"
            )

            # Convert to OpenEndedProblem instances
            for entry in tqdm(filtered_entries, desc=f"Processing {domain}"):
                problem = self._entry_to_problem(entry, domain)
                if problem:
                    all_problems.append(problem)

        logger.major(f"Created {len(all_problems)} total problems from SHP dataset")
        return all_problems

    def _entry_to_problem(self, entry: dict, domain: str) -> Optional[OpenEndedProblem]:
        """
        Convert an SHP entry to an OpenEndedProblem.

        :param entry: SHP data entry
        :param domain: Source domain name
        :return: OpenEndedProblem instance or None if invalid
        """
        # Extract the question/instruction from history field
        question = entry.get("history", "").strip()
        if not question:
            return None

        # Determine which response is preferred (labels: 1 = A preferred, 0 = B preferred)
        if entry.get("labels") == 1:
            ground_truth = entry.get("human_ref_A", "")
        else:
            ground_truth = entry.get("human_ref_B", "")

        if not ground_truth:
            return None

        # Create the problem with all original fields in aux_info
        problem_id = f"{domain}_{entry.get('post_id', 'unknown')}"

        return OpenEndedProblem(
            id=problem_id,
            question=self._transform_question(question),
            aux_info={
                "ground_truth": ground_truth,  # The preferred response
                "domain_name": domain,
                "post_id": entry.get("post_id"),
                "upvote_ratio": entry.get("upvote_ratio"),
                "score_A": entry.get("score_A"),
                "score_B": entry.get("score_B"),
                "human_ref_A": entry.get("human_ref_A"),
                "human_ref_B": entry.get("human_ref_B"),
                "score_ratio": entry.get("score_ratio"),
                "seconds_difference": entry.get("seconds_difference"),
                "created_at_utc_A": entry.get("created_at_utc_A"),
                "created_at_utc_B": entry.get("created_at_utc_B"),
                "c_root_id_A": entry.get("c_root_id_A"),
                "c_root_id_B": entry.get("c_root_id_B"),
                "labels": entry.get("labels"),
            },
        )

    def postprocess_sample(self, sample: Problem) -> Problem:
        """
        Postprocess a sample after it is sampled.
        For OpenEndedProblem, no shuffling is needed.

        :param sample: The sampled problem
        :return: The postprocessed problem
        """
        return sample
