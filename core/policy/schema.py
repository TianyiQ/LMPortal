"""
This file contains the abstract class for a model policy, whether it's an API-based model or local one.
It also contains utility classes, most notably training datasets.
"""

import abc
import asyncio
import copy
import dataclasses
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Union


from core.domain.schema import Problem
from utils.async_utils import run_coroutine
from utils.io_utils import compute_hash, dump_file


@dataclasses.dataclass
class Sample(abc.ABC):
    history: list[dict[str, str]]


@dataclasses.dataclass
class SingleSample(Sample):
    history: list[dict[str, str]]
    output: str
    aux_info: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PairedSample(Sample):
    history: list[dict[str, str]]
    winning_output: str
    losing_output: str
    aux_info: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class EvaluatedSample(Sample):
    history: list[dict[str, str]]
    output: str
    reward: float
    aux_info: dict[str, Any] = dataclasses.field(default_factory=dict)


class Policy(abc.ABC):
    """A model policy. This can be an API-based model, or a local model."""

    colloquial_name: (
        str  # e.g. "GPT-4o", "Llama-3.1-8B-Instruct", "Martingale-trained Llama"
    )
    identifier: str  # unique random string
    few_shot_examples: list[
        dict[str, str]
    ]  # Few-shot examples to prepend to conversations

    @abc.abstractmethod
    def __init__(
        self, colloquial_name: str, few_shot_examples: list[dict[str, str]] = None
    ):
        """Each subclass should implement its own instantiation logic, after calling Policy.__init__() at the beginning."""
        self.colloquial_name = colloquial_name
        self.identifier = hex(random.randint(0, 2**64 - 1))[2:]
        self.few_shot_examples = few_shot_examples or []

    def __str__(self):
        return f"{self.colloquial_name}-ID-{self.identifier}"

    @abc.abstractmethod
    async def infer_from_history_async(
        self,
        history: list[dict[str, str]] | str,
        disable_system_prompt: bool = False,
        **kwargs,
    ) -> str:
        """Each subclass should implement its own async generation method."""
        raise NotImplementedError

    async def embed_async(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        :param texts: List of text strings to embed.
        :type texts: list[str]
        :param kwargs: Additional keyword arguments specific to the embedding model.
        :type kwargs: dict
        :return: List of embedding vectors, one per input text.
        :rtype: list[list[float]]
        """
        raise NotImplementedError(
            f"Embedding not implemented for {self.__class__.__name__}"
        )

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        Synchronous wrapper for embed_async.

        :param texts: List of text strings to embed.
        :type texts: list[str]
        :param kwargs: Additional keyword arguments specific to the embedding model.
        :type kwargs: dict
        :return: List of embedding vectors, one per input text.
        :rtype: list[list[float]]
        """
        return run_coroutine(self.embed_async(texts, **kwargs))

    def infer_from_history(
        self,
        history: list[dict[str, str]] | str,
        disable_system_prompt: bool = False,
        **kwargs,
    ) -> str:
        """
        Given a dialogue history, return a single response.

        :param history: The dialogue history, in OpenAI format.
        :type history: list[dict[str, str]]
        :param disable_system_prompt: Whether to disable the system prompt.
        :type disable_system_prompt: bool
        :param kwargs: Additional keyword arguments, e.g. `disable_reasoning` for local models.
        :type kwargs: dict
        :return: The single response.
        :rtype: str
        """
        return run_coroutine(
            self.infer_from_history_async(history, disable_system_prompt, **kwargs)
        )

    async def infer_async(
        self,
        input_data: Union[list[dict[str, str]], str, Sample, "ProblemDomain"],  # noqa: F821
        disable_system_prompt: bool = False,
        **kwargs,
    ) -> Union[str, SingleSample]:
        """
        Unified async inference method that handles multiple input types.

        :param input_data: Can be:
            - list[dict[str, str]] | str: dialogue history (returns str)
            - Sample: sample to complete (returns SingleSample)
            - ProblemDomain: domain to sample from (returns SingleSample)
        :param disable_system_prompt: Whether to disable the system prompt
        :param kwargs: Additional keyword arguments
        :return: str if input is history, SingleSample if input is Sample or ProblemDomain
        """
        from core.domain.schema import ProblemDomain

        # Handle ProblemDomain: sample one problem and convert to Sample
        if isinstance(input_data, ProblemDomain):
            problems = input_data.sample_problems(n=1)
            input_data = problems[0].to_sample()

        # Handle Sample: extract history, infer, return SingleSample
        if isinstance(input_data, Sample):
            history = input_data.history
            output = await self.infer_from_history_async(
                history, disable_system_prompt, **kwargs
            )
            return SingleSample(
                history=history,
                output=output,
                aux_info=getattr(input_data, "aux_info", {}),
            )

        # Handle standard history input: return str
        return await self.infer_from_history_async(
            input_data, disable_system_prompt, **kwargs
        )

    def infer(
        self,
        input_data: Union[list[dict[str, str]], str, Sample, "ProblemDomain"],  # noqa: F821
        disable_system_prompt: bool = False,
        **kwargs,
    ) -> Union[str, SingleSample]:
        """
        Unified sync inference method that handles multiple input types.

        :param input_data: Can be:
            - list[dict[str, str]] | str: dialogue history (returns str)
            - Sample: sample to complete (returns SingleSample)
            - ProblemDomain: domain to sample from (returns SingleSample)
        :param disable_system_prompt: Whether to disable the system prompt
        :param kwargs: Additional keyword arguments
        :return: str if input is history, SingleSample if input is Sample or ProblemDomain
        """
        return run_coroutine(
            self.infer_async(input_data, disable_system_prompt, **kwargs)
        )

    async def infer_from_histories_async(
        self,
        histories: list[list[dict[str, str]]] | list[str],
        disable_system_prompt: bool = False,
        **kwargs,
    ) -> list[str]:
        """Same as `infer_from_histories`, but async."""
        return await asyncio.gather(
            *[
                self.infer_from_history_async(history, disable_system_prompt, **kwargs)
                for history in histories
            ]
        )

    def infer_from_histories(
        self,
        histories: list[list[dict[str, str]]] | list[str],
        disable_system_prompt: bool = False,
        **kwargs,
    ) -> list[str]:
        """
        Given a list of dialogue histories, return a list of responses.
        By default, this method runs `infer_from_history_async` for each sample individually. You should implement your own `infer_from_histories` if this becomes the perfomance bottleneck, for example if you're using a local model or a batching API.

        :param histories: The list of dialogue histories, in OpenAI format.
        :type histories: list[list[dict[str, str]]]
        :param disable_system_prompt: Whether to disable the system prompt.
        :type disable_system_prompt: bool
        :param kwargs: Additional keyword arguments, e.g. `disable_reasoning` for local models.
        :type kwargs: dict
        :return: The list of responses.
        :rtype: list[str]
        """
        return run_coroutine(
            self.infer_from_histories_async(histories, disable_system_prompt, **kwargs)
        )

    async def infer_many_async(
        self,
        input_data: Union[
            list[list[dict[str, str]]],
            list[str],
            list[Sample],
            tuple["ProblemDomain", int],  # noqa: F821
        ],
        disable_system_prompt: bool = False,
        **kwargs,
    ) -> Union[list[str], list[SingleSample]]:
        """
        Unified async batch inference method that handles multiple input types.

        :param input_data: Can be:
            - list[list[dict[str, str]]] | list[str]: dialogue histories (returns list[str])
            - list[Sample]: samples to complete (returns list[SingleSample])
            - tuple[ProblemDomain, int]: domain and count (returns list[SingleSample])
        :param disable_system_prompt: Whether to disable the system prompt
        :param kwargs: Additional keyword arguments
        :return: list[str] if input is histories, list[SingleSample] if input is list[Sample] or tuple
        """
        from core.domain.schema import ProblemDomain

        # Handle tuple[ProblemDomain, int]: sample n problems
        if isinstance(input_data, tuple) and len(input_data) == 2:
            domain, n = input_data
            if isinstance(domain, ProblemDomain) and isinstance(n, int):
                problems = domain.sample_problems(n=n)
                input_data = [p.to_sample() for p in problems]

        # Handle list[Sample]: process each and return list[SingleSample]
        if input_data and isinstance(input_data[0], Sample):
            results = await asyncio.gather(
                *[
                    self.infer_async(sample, disable_system_prompt, **kwargs)
                    for sample in input_data
                ]
            )
            return results

        # Handle standard histories input: return list[str]
        return await self.infer_from_histories_async(
            input_data, disable_system_prompt, **kwargs
        )

    def infer_many(
        self,
        input_data: Union[
            list[list[dict[str, str]]],
            list[str],
            list[Sample],
            tuple["ProblemDomain", int],  # noqa: F821
        ],
        disable_system_prompt: bool = False,
        **kwargs,
    ) -> Union[list[str], list[SingleSample]]:
        """
        Unified sync batch inference method that handles multiple input types.

        :param input_data: Can be:
            - list[list[dict[str, str]]] | list[str]: dialogue histories (returns list[str])
            - list[Sample]: samples to complete (returns list[SingleSample])
            - tuple[ProblemDomain, int]: domain and count (returns list[SingleSample])
        :param disable_system_prompt: Whether to disable the system prompt
        :param kwargs: Additional keyword arguments
        :return: list[str] if input is histories, list[SingleSample] if input is list[Sample] or tuple
        """
        return run_coroutine(
            self.infer_many_async(input_data, disable_system_prompt, **kwargs)
        )

    def supports_logprobs(self) -> bool:
        """
        Whether this policy supports logprobs. Override this method if your policy supports it.
        """
        return False

    def logprobs_single(
        self, dialogue: list[dict[str, str]], return_summed: bool = True, **kwargs
    ) -> float | list[float]:
        """Each subclass should implement its own async logprob calculation method.

        :param dialogue: OpenAI-format dialogue, e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        :type dialogue: list[dict[str, str]]
        :param return_summed: If True, return sum of logprobs. If False, return list of per-token logprobs.
        :type return_summed: bool
        :param kwargs: Additional keyword arguments, e.g. `disable_reasoning` for local models.
        :type kwargs: dict
        :return: Sum of log probabilities for the dialogue (if return_summed=True) or list of per-token logprobs (if return_summed=False)
        :rtype: float | list[float]
        """
        return run_coroutine(
            self.logprobs_single_async(dialogue, return_summed, **kwargs)
        )

    @abc.abstractmethod
    async def logprobs_single_async(
        self, dialogue: list[dict[str, str]], return_summed: bool = True, **kwargs
    ) -> float | list[float]:
        """Each subclass should implement its own async logprob calculation method.

        :param dialogue: OpenAI-format dialogue, e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        :type dialogue: list[dict[str, str]]
        :param return_summed: If True, return sum of logprobs. If False, return list of per-token logprobs.
        :type return_summed: bool
        :param kwargs: Additional keyword arguments, e.g. `disable_reasoning` for local models.
        :type kwargs: dict
        :return: Sum of log probabilities for the dialogue (if return_summed=True) or list of per-token logprobs (if return_summed=False)
        :rtype: float | list[float]
        """
        raise NotImplementedError

    def logprobs_batch(
        self,
        dialogues: list[list[dict[str, str]]],
        return_summed: bool = True,
        **kwargs,
    ) -> list[float] | list[list[float]]:
        """Calculate sum of logprobs for a batch of dialogues. Default implementation uses individual calls.

        :param dialogues: List of OpenAI-format dialogues
        :type dialogues: list[list[dict[str, str]]]
        :param return_summed: If True, return sums of logprobs. If False, return lists of per-token logprobs.
        :type return_summed: bool
        :param kwargs: Additional keyword arguments, e.g. `disable_reasoning` for local models.
        :type kwargs: dict
        :return: List of sum of log probabilities for each dialogue (if return_summed=True) or list of per-token logprob lists (if return_summed=False)
        :rtype: list[float] | list[list[float]]
        """
        return run_coroutine(
            self.logprobs_batch_async(dialogues, return_summed, **kwargs)
        )

    async def logprobs_batch_async(
        self,
        dialogues: list[list[dict[str, str]]],
        return_summed: bool = True,
        **kwargs,
    ) -> list[float] | list[list[float]]:
        """Calculate sum of logprobs for a batch of dialogues. Default implementation uses individual calls.

        :param dialogues: List of OpenAI-format dialogues
        :type dialogues: list[list[dict[str, str]]]
        :param return_summed: If True, return sums of logprobs. If False, return lists of per-token logprobs.
        :type return_summed: bool
        :param kwargs: Additional keyword arguments, e.g. `disable_reasoning` for local models.
        :type kwargs: dict
        :return: List of sum of log probabilities for each dialogue (if return_summed=True) or list of per-token logprob lists (if return_summed=False)
        :rtype: list[float] | list[list[float]]
        """
        # Default implementation - subclasses can override for true batch processing
        tasks = [
            self.logprobs_single_async(dialogue, return_summed=return_summed)
            for dialogue in dialogues
        ]
        return await asyncio.gather(*tasks)

    def train_sft(
        self,
        samples: list[SingleSample],
        validation_samples: list[SingleSample] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "Policy":
        """Perform SFT training. Each subclass should implement this by itself, if supported. This method is out-of-place."""
        return run_coroutine(
            self.train_sft_async(samples, validation_samples, metadata)
        )

    async def train_sft_async(
        self,
        samples: list[SingleSample],
        validation_samples: list[SingleSample] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "Policy":
        """Perform async SFT training. Each subclass should implement this by itself, if supported. This method is out-of-place."""
        raise NotImplementedError

    def train_dpo(
        self,
        samples: list[PairedSample],
        validation_samples: list[PairedSample | SingleSample] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "Policy":
        """Perform DPO training. Each subclass should implement this by itself, if supported. This method is out-of-place."""
        return run_coroutine(
            self.train_dpo_async(samples, validation_samples, metadata)
        )

    async def train_dpo_async(
        self,
        samples: list[PairedSample],
        validation_samples: list[PairedSample | SingleSample] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "Policy":
        """Perform async DPO training. Each subclass should implement this by itself, if supported. This method is out-of-place."""
        raise NotImplementedError

    def train_rl(
        self,
        samples: list[Problem | EvaluatedSample],
        grader=None,
        validation_samples: list[Problem | EvaluatedSample | SingleSample] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "Policy":
        """Perform offline PPO training with pre-determined reward values. Each subclass should implement this by itself, if supported. This method is out-of-place."""
        return run_coroutine(
            self.train_rl_async(samples, grader, validation_samples, metadata)
        )

    async def train_rl_async(
        self,
        samples: list[Problem | EvaluatedSample],
        grader=None,
        validation_samples: list[Problem | EvaluatedSample | SingleSample] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "Policy":
        """Perform async offline PPO training with pre-determined reward values. Each subclass should implement this by itself, if supported. This method is out-of-place."""
        raise NotImplementedError

    def deep_copy(
        self,
        suffix_type: str,
        suffix_data: Any = None,
        metadata: dict = None,
        characteristics: str = "",
        **kwargs,
    ) -> "Policy":
        """
        Create a deep copy of this policy with updated name, identifier, and saved metadata.
        This is a utility method used by training/few-shot methods.

        :param suffix_type: Type suffix for the new name (e.g., "sft", "fewshot", "dpo")
        :param suffix_data: Data to hash for unique suffix (if None, uses timestamp)
        :param metadata: Additional metadata to save
        :param kwargs: Additional attributes to set on the new policy
        :return: New policy instance with updated properties
        """
        # Create deep copy
        new_policy = copy.deepcopy(self)

        # Generate unique suffix
        if suffix_data is not None:
            suffix_hash = compute_hash(suffix_data, length=12)
        else:
            suffix_hash = compute_hash(datetime.now().isoformat(), length=8)

        # Update name and identifier
        date_today = datetime.now().strftime("%y%m%d")
        characteristics = characteristics or metadata.get("characteristics", "")
        new_policy.colloquial_name = f"{self.colloquial_name}-{suffix_type}-{characteristics}-{date_today}-{suffix_hash}"
        new_policy.identifier = hex(random.randint(0, 2**64 - 1))[2:]

        # Set any additional attributes
        for key, value in kwargs.items():
            setattr(new_policy, key, value)

        # Save metadata to model directory
        model_dir = Path("data/models") / new_policy.colloquial_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Combine all metadata
        full_metadata = {
            "base_model": self.colloquial_name,
            "suffix_type": suffix_type,
            "timestamp": datetime.now().isoformat(),
            "original_identifier": self.identifier,
        }
        if metadata:
            full_metadata.update(metadata)

        # Save metadata
        dump_file(model_dir / "metadata.json", full_metadata, indent=2, default=str)

        return new_policy

    def add_few_shot_examples(
        self,
        few_shot_examples: list[dict[str, str]],
        metadata: dict = {},  # noqa: B006
    ) -> "Policy":
        """
        Add few-shot examples to create a new policy (out-of-place).

        :param few_shot_examples: List of dialogue turns to use as few-shot examples
        :return: New policy instance with few-shot examples
        """
        # Use deep_copy to create new instance with proper naming
        new_policy = self.deep_copy(
            suffix_type="fewshot",
            suffix_data=few_shot_examples,
            metadata={
                "num_samples": len((self.few_shot_examples or []) + few_shot_examples)
                // 2,
                **metadata,
            },  # Assuming pairs
            few_shot_examples=(self.few_shot_examples or []) + few_shot_examples,
        )

        # Save few-shot examples to model directory
        model_dir = Path("data/models") / new_policy.colloquial_name
        examples_file = model_dir / "few_shot_examples.json"
        dump_file(examples_file.as_posix(), few_shot_examples, indent=2)

        return new_policy

    def _prepend_few_shot_to_history(
        self, history: list[dict[str, str]] | str
    ) -> list[dict[str, str]]:
        """
        Prepend few-shot examples to a conversation history.

        :param history: The conversation history
        :return: History with few-shot examples prepended
        """
        # Convert string history to list format
        if isinstance(history, str):
            history = [{"role": "user", "content": history}]

        if not self.few_shot_examples:
            return history

        # Prepend few-shot examples after system message (if any)
        result = []
        system_messages = [msg for msg in history if msg.get("role") == "system"]
        other_messages = [msg for msg in history if msg.get("role") != "system"]

        # Add system messages first
        result.extend(system_messages)

        # Add few-shot examples
        result.extend(self.few_shot_examples)

        # Add the actual conversation
        result.extend(other_messages)

        return result

    def train(
        self,
        samples: list[Sample],
        validation_samples: list[Sample] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "Policy":
        """Perform training out-of-place."""
        return run_coroutine(self.train_async(samples, validation_samples, metadata))

    async def train_async(
        self,
        samples: list[Sample],
        validation_samples: list[Sample] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "Policy":
        """Perform async training out-of-place."""
        if not samples:
            return self

        if isinstance(samples[0], PairedSample):
            return await self.train_dpo_async(
                samples, validation_samples=validation_samples, metadata=metadata
            )
        elif isinstance(samples[0], SingleSample):
            return await self.train_sft_async(
                samples, validation_samples=validation_samples, metadata=metadata
            )
        elif isinstance(samples[0], EvaluatedSample):
            return await self.train_rl_async(
                samples, validation_samples=validation_samples, metadata=metadata
            )
        else:
            raise TypeError(
                "Unrecognized sample type. Must be list[PairedSample] or list[SingleSample] or list[EvaluatedSample]."
            )
