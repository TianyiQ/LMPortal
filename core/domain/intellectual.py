import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from core.domain.schema import OpenEndedProblem, Problem, ProblemDomain
from utils.io_utils import logger


class IntellectualDemonstration(ProblemDomain):
    """
    Intellectual Demonstration domain using diverse high-quality Q&A sources.

    This domain supports loading from multiple unified Q&A data files with
    consistent field naming. Supports both list[str] and dict[str, int] for
    specifying data sources with optional sampling limits.
    """

    # Default data files - can be overridden in constructor
    DEFAULT_DATA_FILES = [
        "data/questions/cmv_qa.json",  # 4k
        "data/questions/institutional_books_qa.json",  # 2k
        "data/questions/new_papers_qa.json",  # 20k
    ]

    # Source categories for balanced sampling with fine-grained details
    SOURCE_CATEGORIES = {
        "arxiv": ["arxiv", "arxiv_papers", "papers_qa"],
        "books": ["institutional_books", "inst_books"],
        "intellectual": ["intellectual", "intellectual_elite"],
        "philosophy": ["philosophy", "stanford_encyclopedia", "philpapers"],
        "stackexchange": ["stackexchange", "stackoverflow"],
        "academic": ["pubmed", "wikipedia", "google_scholar", "lesswrong"],
        "legal": ["supreme_court", "oyez_supreme_court"],
        "other": [],  # Catch-all category
    }

    def __init__(
        self,
        data_files: Optional[Union[list[str], dict[str, int]]] = None,
        train_size: float = 0.8,
        max_samples_per_source: int = 5000,
        min_answer_length: int = 50,
        exclude_memory_based: bool = True,
        balanced_sampling: bool = False,
        random_seed: Optional[int] = None,
    ):
        """
        Instantiate an IntellectualDemonstration problem set with unified data.

        :param data_files: Either a list of file paths to load all data from, or
                          a dict mapping file paths to the number of samples to
                          take from each file (random sampling without replacement)
        :type data_files: Optional[Union[List[str], Dict[str, int]]]
        :param train_size: Portion of samples for training split (default: 0.8)
        :type train_size: float
        :param max_samples_per_source: Maximum samples per source to prevent dominance
        :type max_samples_per_source: int
        :param min_answer_length: Minimum answer length to filter out low-quality
        :type min_answer_length: int
        :param exclude_memory_based: Whether to exclude memory-based questions
        :type exclude_memory_based: bool
        :param balanced_sampling: Whether to balance across source categories
        :type balanced_sampling: bool
        :param random_seed: Random seed for reproducibility
        :type random_seed: Optional[int]
        """
        super().__init__()
        self.train_size = train_size
        self.max_samples_per_source = max_samples_per_source
        self.min_answer_length = min_answer_length
        self.exclude_memory_based = exclude_memory_based
        self.balanced_sampling = balanced_sampling

        # Set random seed if provided
        if random_seed is not None:
            self._rng = np.random.RandomState(random_seed)

        # Use provided files or default
        if data_files is None:
            data_files = self.DEFAULT_DATA_FILES

        # Load all data
        self.questions_all = self._load_all_unified_data(data_files)

        # Log statistics
        logger.major(f"Loaded {len(self.questions_all)} Q&A pairs total")

        # Create train/test splits
        self.make_questions_splits(self.train_size)

    def _load_all_unified_data(
        self, data_files: Union[list[str], dict[str, int]]
    ) -> list[Problem]:
        """
        Load Q&A data from unified format files.

        :param data_files: Either list of paths or dict of path->sample_count
        :return: List of OpenEndedProblem instances
        """
        all_problems = []
        sources_count = {}

        # Convert to dict format for uniform processing
        if isinstance(data_files, list):
            files_dict = {f: None for f in data_files}  # None means take all
        else:
            files_dict = data_files

        for file_path, sample_limit in files_dict.items():
            path = Path(file_path)
            if not path.exists():
                logger.major(f"Warning: File not found: {file_path}, skipping...")
                continue

            logger.major(f"Loading from {file_path}...")

            try:
                with open(path) as f:
                    data = json.load(f)

                qa_pairs = data.get("qa_pairs", [])

                # Apply sampling if limit specified
                if sample_limit is not None and sample_limit < len(qa_pairs):
                    logger.major(
                        f"  Sampling {sample_limit} from {len(qa_pairs)} available"
                    )
                    qa_pairs = qa_pairs[:sample_limit]
                    # indices = self._rng.choice(
                    #     len(qa_pairs), size=sample_limit, replace=False
                    # )
                    # qa_pairs = [qa_pairs[i] for i in indices]

                skipped_memory = 0
                skipped_short = 0
                processed = 0

                for qa in qa_pairs:
                    # Check memory-based field (handle both formats)
                    is_memory = qa.get("memory_based", qa.get("memory-based", False))

                    # Skip memory-based if configured
                    if self.exclude_memory_based and is_memory:
                        skipped_memory += 1
                        continue

                    # Extract fields
                    question = qa.get("question", "").strip()
                    answer = qa.get("answer", "").strip()
                    source = qa.get("source", "unknown")

                    # Skip if missing required fields
                    if not question or not answer:
                        continue

                    # Skip short answers
                    if len(answer) < self.min_answer_length:
                        skipped_short += 1
                        continue

                    # Check source limits
                    if source in sources_count:
                        if sources_count[source] >= self.max_samples_per_source:
                            continue

                    # Create problem instance
                    problem = OpenEndedProblem(
                        id=qa.get("id", f"{source}_{len(all_problems)}"),
                        question=question,
                        aux_info={
                            "answer": answer,
                            "ground_truth": answer,  # Include both for compatibility
                            "source": source,
                            "source_url": qa.get("source_url", ""),
                            "metadata": qa.get("metadata", {}),
                            "memory_based": is_memory,
                        },
                    )

                    all_problems.append(problem)
                    sources_count[source] = sources_count.get(source, 0) + 1
                    processed += 1

                logger.major(f"  Processed {processed} problems from {file_path}")
                if skipped_memory > 0:
                    logger.major(f"  Skipped {skipped_memory} memory-based questions")
                if skipped_short > 0:
                    logger.major(f"  Skipped {skipped_short} short answers")

            except Exception as e:
                logger.major(f"Error loading {file_path}: {e}")
                continue

        # Apply balanced sampling if requested
        if self.balanced_sampling:
            all_problems = self._apply_balanced_sampling(all_problems)

        # Log source distribution
        logger.major("\nSource distribution:")
        for source, count in sorted(
            sources_count.items(), key=lambda x: x[1], reverse=True
        )[:20]:
            logger.major(f"  {source}: {count}")

        return all_problems

    def _apply_balanced_sampling(self, problems: list[Problem]) -> list[Problem]:
        """
        Apply balanced sampling across source categories.

        :param problems: All loaded problems
        :return: Balanced sample of problems
        """
        # Group by category
        categorized = {}
        for problem in problems:
            source = problem.aux_info.get("source", "unknown")
            category = self._get_source_category(source)

            if category not in categorized:
                categorized[category] = []
            categorized[category].append(problem)

        # Sample evenly from each category
        balanced = []
        max_per_category = len(problems) // len(categorized) if categorized else 1000

        for category, cat_problems in categorized.items():
            sample_size = min(len(cat_problems), max_per_category)
            if sample_size < len(cat_problems):
                sampled = cat_problems[:sample_size]
                # Use numpy random choice for sampling
                # indices = self._rng.choice(
                #     len(cat_problems), size=sample_size, replace=False
                # )
                # sampled = [cat_problems[i] for i in indices]
            else:
                sampled = cat_problems
            balanced.extend(sampled)
            logger.major(
                f"  Category {category}: sampled {sample_size} from {len(cat_problems)}"
            )

        # Shuffle using RNG
        self._rng.shuffle(balanced)
        return balanced

    def _get_source_category(self, source: str) -> str:
        """
        Get the category for a source.

        :param source: Source name
        :return: Category name
        """
        source_lower = source.lower()
        for category, patterns in self.SOURCE_CATEGORIES.items():
            for pattern in patterns:
                if pattern in source_lower:
                    return category

        # Default category for uncategorized sources
        return "other"

    def postprocess_sample(self, sample: Problem) -> Problem:
        """
        Postprocess a sample after it is sampled.
        For OpenEndedProblem, we ensure the answer is in the expected format.

        :param sample: The sampled problem
        :return: The postprocessed problem
        """
        # Ensure the ground_truth field is set if not already
        if "ground_truth" not in sample.aux_info and "answer" in sample.aux_info:
            sample.aux_info["ground_truth"] = sample.aux_info["answer"]

        return sample

    def get_sample_problems(self, n: int = 5) -> list[tuple[str, str, str]]:
        """
        Get sample problems for display.

        :param n: Number of samples to return
        :return: List of (question, answer, source) tuples
        """
        samples = []
        # Use RNG for sampling
        num_samples = min(n, len(self.questions_all))
        indices = self._rng.choice(
            len(self.questions_all), size=num_samples, replace=False
        )
        problems = [self.questions_all[i] for i in indices]

        for problem in problems:
            question = problem.question
            answer = problem.aux_info.get(
                "answer", problem.aux_info.get("ground_truth", "No answer")
            )
            source = problem.aux_info.get("source", "unknown")
            samples.append((question, answer, source))

        return samples
