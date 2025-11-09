import asyncio
import concurrent
import dataclasses
import json
import os
import random
import warnings
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

import aiohttp

try:
    from safetytooling.apis import InferenceAPI
except ImportError:
    print("WARNING: SafetyTooling API loading failed, will not use it.")
    InferenceAPI = None

try:
    from safetytooling.data_models import (
        ChatMessage,
        LLMResponse,
        MessageRole,
        Prompt,
        StopReason,
    )
except ImportError:
    print("WARNING: SafetyTooling not found, APIModel will not work.")
    ChatMessage = None
    LLMResponse = None
    MessageRole = None
    Prompt = None
    StopReason = None

from core.domain.schema import Problem
from core.policy.schema import Policy, SingleSample
from utils.io_utils import dump_file, logger


class APIModel(Policy):
    """A model that is hosted on an API."""

    model_provider: str
    model_name: str
    system_prompt: str | None

    # shared API instance and across all instances of this class
    shared_api = (
        InferenceAPI(  # type: ignore
            together_num_threads=40,
            anthropic_num_threads=40,
            openai_num_threads=5000,
            openai_s2s_num_threads=5000,
            gpt4o_s2s_rpm_cap=30000,
            no_cache=True,
            prompt_history_dir=None,
        )
        if InferenceAPI is not None
        else None
    )

    # shared parameters for all instances of this class
    temperature = float(os.getenv("TEMPERATURE", 0.25))
    presence_penalty = float(os.getenv("PRESENCE_PENALTY", 0.0))

    def __init__(
        self,
        model_name: str,
        colloquial_name: str | None = None,
        system_prompt: str | None = None,
        model_provider: Literal[
            "auto",
            "openai",
            "anthropic",
            "google",
            "together",
            "deepseek",
            "openrouter",
        ] = "auto",
        few_shot_examples: list[dict[str, str]] = None,
        response_only: bool = True,  # Unused, kept for compatibility
    ):
        """
        Instantiate an API model.

        :param model_name: The name of the model, as it appears in the provider's API.
        :type model_name: str
        :param colloquial_name: The colloquial name of the model, defaults to `model_name`.
        :type colloquial_name: str | None
        :param model_provider: The provider of the model.
        :type model_provider: Literal["openai", "anthropic", "google", "together", "deepseek", "openrouter"]
        :param few_shot_examples: Few-shot examples to prepend to conversations
        :type few_shot_examples: list[dict[str, str]] | None
        """
        super().__init__(
            colloquial_name=colloquial_name or model_name,
            few_shot_examples=few_shot_examples,
        )
        self.model_provider = model_provider
        self.model_name = model_name
        self.system_prompt = system_prompt

        if "gemini" in self.model_name or "google" in self.model_provider:
            # Dial up temperature to avoid RECITATION errors
            self.temperature = 1.0

    async def infer_from_history_async(
        self,
        history: list[dict[str, str]] | str,
        disable_system_prompt: bool = False,
        disable_logging: bool = True,
        **kwargs,
    ) -> str:
        """
        Given a dialogue history, return a single response.
        By implementing this method, the `infer_from_history` and `infer_from_histories` methods will automatically be available for use.
        You may pass additional arguments to the API call, for example `infer_from_history(..., temperature=0.25, is_valid=lambda s: len(s) > 5)`.

        :param history: The dialogue history, in OpenAI format.
        :type history: list[dict[str, str]]
        :return: The single response.
        :rtype: str
        """
        if isinstance(history, str):
            history = [{"role": "user", "content": history}]
        else:
            history = deepcopy(history)

        # Prepend few-shot examples if available
        if self.few_shot_examples:
            history = self._prepend_few_shot_to_history(history)

        if (
            not disable_system_prompt
            and self.system_prompt
            and not any(msg.get("role") == "system" for msg in history)
        ):
            history.insert(0, {"role": "system", "content": self.system_prompt})

        if not disable_logging:
            dump_file(
                "data/tmp/___inference_record.jsonl",
                history,
                write_mode="a",
                indent=None,
            )

        prompt = Prompt(
            messages=[
                ChatMessage(
                    content=message["content"], role=MessageRole(message["role"])
                )
                for message in history
            ]
        )

        if "temperature" not in kwargs:
            kwargs["temperature"] = self.temperature
        if "presence_penalty" not in kwargs:
            kwargs["presence_penalty"] = self.presence_penalty

        if "o3" in self.model_name or "o4" in self.model_name:
            del kwargs["temperature"]

        if "claude" in self.model_name and "presence_penalty" in kwargs:
            if float(kwargs["presence_penalty"]) != 0:
                warnings.warn(
                    "Claude API doesn't support presence penalty. Ignoring the penalty."
                )

            del kwargs["presence_penalty"]

        if "seed" not in kwargs:
            kwargs["seed"] = random.randint(0, 2**30 - 1)

        if self.model_provider != "auto":
            kwargs["force_provider"] = self.model_provider

        # Check the correctness of the dialogue structure
        for msg_idx in range(1, len(prompt.messages)):
            this_msg, last_msg = prompt.messages[msg_idx], prompt.messages[msg_idx - 1]
            is_valid = (
                this_msg.role.value
                in ("user", "assistant")  # No system prompt after the first entry
                and this_msg.role.value
                != last_msg.role.value  # No two consecutive speeches from same person
            )
            if not is_valid:
                print(f"Malstructured query: {prompt}")
                raise ValueError("Malstructured LLM query.")

        if int(os.getenv("NO_RETRY", "0")):
            kwargs["max_attempts_per_api_call"] = 1

        response: list[LLMResponse] = await self.shared_api(
            model_id=self.model_name,
            prompt=prompt,
            print_prompt_and_response=False,
            **kwargs,
        )

        try:
            res = response[0].completion
        except Exception:
            res = response[0][0].completion

        if not disable_logging:
            dump_file(
                "data/tmp/___inference_record.jsonl",
                res,
                write_mode="a",
                indent=None,
            )
        return res

    def supports_logprobs(self) -> bool:
        """
        Whether this policy supports logprobs. Override this method if your policy supports it.
        """
        return self.model_provider in ["together"]

    async def _logprobs_together(
        self, dialogue: list[dict[str, str]], return_summed: bool = True, **kwargs
    ) -> float | list[float]:
        """
        Calculate the log probabilities for the given dialogue using Together's completion API.

        This method directly calls Together's completion API with echo=True to get logprobs
        for the entire conversation including both prompt and response.
        """
        # Get Together API key
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TOGETHER_API_KEY not found. Please set it in environment or lib/safety_tooling/.env"
            )

        # Convert dialogue to completion format
        # Together's completion API expects a single prompt string
        prompt_parts = []

        # Add system prompt if configured
        history = deepcopy(dialogue)
        if self.system_prompt and (not history or history[0].get("role") != "system"):
            history.insert(0, {"role": "system", "content": self.system_prompt})

        # Format conversation as a completion prompt
        # Using a format that mimics chat structure but works with completion API
        for msg in history:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"System: {content}\n\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n\n")

        # Join all parts into a single prompt
        prompt = "".join(prompt_parts)

        # If the last message was from user, we need to add "Assistant:" to trigger completion
        if history and history[-1]["role"] == "user":
            prompt += "Assistant:"

        # Prepare the API request
        url = "https://api.together.xyz/v1/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Request parameters for Together completion API
        # When echo=True, we get logprobs for the entire prompt plus any generated tokens

        # For conversations ending with assistant message, we don't want extra generation
        # For those ending with user message, we might want a small amount
        if history and history[-1]["role"] == "assistant":
            max_new_tokens = 1  # Minimal generation
        else:
            max_new_tokens = 10  # Allow small generation after user message

        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,  # Limit new generation
            "temperature": 0.0,  # Deterministic
            "echo": True,  # Critical: return prompt + completion with logprobs
            "logprobs": 1,  # Return top 1 logprob per token
            "stop": [
                "<|endoftext|>",
                "<|eot_id|>",
                "\n\n",
                "User:",
                "Assistant:",
            ],  # Stop sequences
        }

        # Make the API request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Together API error (status {response.status}): {error_text}"
                        )

                    result = await response.json()

                    # Debug: Print response structure if needed
                    if kwargs.get("debug", False):
                        print("DEBUG: Together API Response:")
                        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])

                    # Extract logprobs from the response
                    # With echo=True, the prompt logprobs are in result["prompt"][0]["logprobs"]
                    # and any generated tokens are in result["choices"][0]["logprobs"]

                    all_token_logprobs = []
                    all_tokens = []

                    # First, get prompt logprobs (these contain the echoed input)
                    if "prompt" in result and result["prompt"]:
                        prompt_data = result["prompt"][0]
                        if "logprobs" in prompt_data:
                            prompt_logprobs = prompt_data["logprobs"]
                            if "token_logprobs" in prompt_logprobs:
                                all_token_logprobs.extend(
                                    prompt_logprobs["token_logprobs"]
                                )
                            if "tokens" in prompt_logprobs:
                                all_tokens.extend(prompt_logprobs["tokens"])

                    # Then, get any generated token logprobs
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        if "logprobs" in choice:
                            choice_logprobs = choice["logprobs"]
                            if "token_logprobs" in choice_logprobs:
                                all_token_logprobs.extend(
                                    choice_logprobs["token_logprobs"]
                                )
                            if "tokens" in choice_logprobs:
                                all_tokens.extend(choice_logprobs["tokens"])

                    if not all_token_logprobs:
                        raise ValueError(
                            "No token_logprobs found in Together API response"
                        )

                    # Filter out None values (first token typically has None)
                    valid_logprobs = []
                    valid_tokens = []
                    for i, lp in enumerate(all_token_logprobs):
                        if lp is not None:
                            valid_logprobs.append(lp)
                            if i < len(all_tokens):
                                valid_tokens.append(all_tokens[i])
                            else:
                                valid_tokens.append(f"<token_{i}>")

                    if not valid_logprobs:
                        raise ValueError("No valid logprobs found (all were None)")

                    # Return based on return_summed parameter
                    if return_summed:
                        return float(sum(valid_logprobs))
                    else:
                        # Return just the logprobs without tokens to match schema
                        return [float(lp) for lp in valid_logprobs]

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error calling Together API: {e}") from e
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse Together API response: {e}") from e

    async def logprobs_single_async(
        self, dialogue: list[dict[str, str]], return_summed: bool = True, **kwargs
    ) -> float | list[float]:
        """
        Calculate the log probabilities for the given dialogue.

        For API models, this uses the provided dialogue and requests log probabilities
        from the API.

        :param dialogue: OpenAI-format dialogue, e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        :type dialogue: list[dict[str, str]]
        :param return_summed: If True, return sum of logprobs. If False, return list of per-token logprobs.
        :type return_summed: bool
        :return: Sum of log probabilities (if return_summed=True) or list of per-token logprobs (if return_summed=False)
        :rtype: float | list[float]
        """
        dialogue = self._prepend_few_shot_to_history(dialogue)
        if self.model_provider == "together":
            return await self._logprobs_together(dialogue, return_summed, **kwargs)
        else:
            raise NotImplementedError(
                f"Logprobs are not supported for {self.model_provider} models."
            )

    async def train_sft_async(
        self,
        samples: list[SingleSample],
        validation_samples: list[SingleSample] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "APIModel":
        """
        Perform supervised fine-tuning using OpenAI or Together AI APIs.

        :param samples: List of SingleSample training examples
        :param validation_samples: Optional list of validation samples
        :param metadata: Training metadata
        :return: New fine-tuned APIModel instance
        """
        # Track if WandB is available for syncing
        use_wandb = False
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key and self.model_provider == "openai":
            try:
                from wandb.integration.openai.fine_tuning import WandbLogger

                use_wandb = True
            except ImportError:
                logger.urgent(
                    "WandB OpenAI integration not found. Install with: pip install wandb"
                )
                use_wandb = False

        # Only OpenAI and Together support fine-tuning
        if self.model_provider not in ["openai", "together"]:
            raise NotImplementedError(
                f"Fine-tuning not supported for provider '{self.model_provider}'. "
                "Only 'openai' and 'together' providers support fine-tuning."
            )

        # Create a deep copy early to get the colloquial_name for logging
        timestamp = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(0, 1000000)}"
        )
        temp_model = self.deep_copy(
            suffix_type="sft", suffix_data=samples, metadata=metadata
        )
        model_display_name = (
            temp_model.colloquial_name[:30] + "..."
            if len(temp_model.colloquial_name) > 30
            else temp_model.colloquial_name
        )

        # Prepare training data in JSONL format
        training_data = []
        for sample in samples:
            # Format as OpenAI chat format
            messages = sample.history + [
                {"role": "assistant", "content": sample.output}
            ]
            training_data.append({"messages": messages})

        # Save training data to temporary file
        training_file = Path("data/tmp") / f"training_data_{timestamp}.jsonl"
        training_file.parent.mkdir(parents=True, exist_ok=True)

        with open(training_file, "w", encoding="utf-8") as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.major(
            f"[{model_display_name}] Saved {len(training_data)} training examples to {training_file}"
        )

        # Prepare validation data if provided
        validation_file = None
        if validation_samples:
            validation_data = []
            for sample in validation_samples:
                messages = sample.history + [
                    {"role": "assistant", "content": sample.output}
                ]
                validation_data.append({"messages": messages})

            validation_file = Path("data/tmp") / f"validation_data_{timestamp}.jsonl"
            with open(validation_file, "w", encoding="utf-8") as f:
                for item in validation_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            logger.major(
                f"[{model_display_name}] Saved {len(validation_data)} validation examples to {validation_file}"
            )

        try:
            if self.model_provider == "openai":
                # OpenAI fine-tuning
                from openai import OpenAI

                client = OpenAI()

                # Upload training file
                with open(training_file, "rb") as f:
                    file_response = client.files.create(file=f, purpose="fine-tune")

                logger.major(
                    f"[{model_display_name}] Uploaded training file: {file_response.id}"
                )

                # Upload validation file if provided
                validation_file_id = None
                if validation_file:
                    with open(validation_file, "rb") as f:
                        val_file_response = client.files.create(
                            file=f, purpose="fine-tune"
                        )
                    validation_file_id = val_file_response.id
                    logger.major(
                        f"[{model_display_name}] Uploaded validation file: {validation_file_id}"
                    )

                # Create fine-tuning job with optional validation
                job_params = {
                    "training_file": file_response.id,
                    "model": self.model_name,
                    "suffix": f"sft-{timestamp[:8]}",
                    "n_epochs": metadata.get("num_epochs", 1),
                }
                if validation_file_id:
                    job_params["validation_file"] = validation_file_id

                fine_tune_job = client.fine_tuning.jobs.create(**job_params)

                logger.urgent(
                    f"[{model_display_name}] Created SFT job: {fine_tune_job.id}"
                )

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Sync with WandB if available
                    if use_wandb:
                        try:
                            from wandb.integration.openai.fine_tuning import WandbLogger

                            event_loop = asyncio.get_event_loop()
                            event_loop.run_in_executor(
                                executor, WandbLogger.sync, fine_tune_job.id
                            )
                            logger.urgent(
                                f"[{model_display_name}] WandB syncing enabled for job: {fine_tune_job.id}"
                            )
                        except Exception as e:
                            logger.urgent(f"Failed to sync with WandB: {e}")

                    # Wait for fine-tuning to complete with status updates
                    status_counter = 0
                    while True:
                        job_status = client.fine_tuning.jobs.retrieve(fine_tune_job.id)

                        # Print status update every 3rd iteration (reduced frequency)
                        if status_counter % 3 == 0:
                            status_msg = f"[{model_display_name}] Fine-tuning status: {job_status.status}"

                            # Add validation loss if available from OpenAI's training events
                            if job_status.status == "running":
                                try:
                                    # Get latest training metrics from events
                                    events = list(
                                        client.fine_tuning.jobs.list_events(
                                            fine_tuning_job_id=fine_tune_job.id,
                                            limit=10,
                                        )
                                    )
                                    for event in events:
                                        if hasattr(event, "data") and event.data:
                                            if "train_loss" in event.data:
                                                status_msg += f" | Train Loss: {event.data['train_loss']:.4f}"
                                            if "valid_loss" in event.data:
                                                status_msg += f" | Val Loss: {event.data['valid_loss']:.4f}"
                                            break  # Use first event with metrics
                                except Exception:
                                    pass  # Ignore errors getting events

                            logger.urgent(status_msg)

                        status_counter += 1

                        if job_status.status == "succeeded":
                            break
                        elif job_status.status == "failed":
                            raise RuntimeError(
                                f"Fine-tuning failed: {job_status.error}"
                            )

                        await asyncio.sleep(30)

                # Get the fine-tuned model name
                fine_tuned_model_name = job_status.fine_tuned_model

            elif self.model_provider == "together":
                # Together AI fine-tuning
                from together import Together

                client = Together()

                # Upload training file
                file_response = client.files.upload(file=training_file)

                logger.major(
                    f"[{model_display_name}] Uploaded training file: {file_response.id}"
                )

                # Upload validation file if provided
                validation_file_id = None
                if validation_file:
                    val_file_response = client.files.upload(file=validation_file)
                    validation_file_id = val_file_response.id
                    logger.major(
                        f"[{model_display_name}] Uploaded validation file: {validation_file_id}"
                    )

                # Create fine-tuning job with optional validation
                job_params = {
                    "training_file": file_response.id,
                    "model": self.model_name,
                    "n_epochs": metadata.get("num_epochs", 1),
                    "batch_size": "max",
                    "learning_rate": 1e-5,
                    "suffix": f"sft-{timestamp[:8]}",
                }
                if validation_file_id:
                    job_params["validation_file"] = validation_file_id

                fine_tune_job = client.fine_tuning.create(**job_params)

                logger.major(
                    f"[{model_display_name}] Created fine-tuning job: {fine_tune_job.id}"
                )

                # Wait for fine-tuning to complete with status updates
                status_counter = 0
                while True:
                    job_status = client.fine_tuning.retrieve(id=fine_tune_job.id)

                    # Print status update every 3rd iteration (reduced frequency)
                    if status_counter % 3 == 0:
                        status_msg = f"[{model_display_name}] Fine-tuning status: {job_status.status}"

                        # Add metrics if available (Together AI may have different attribute names)
                        if (
                            hasattr(job_status, "training_metrics")
                            and job_status.training_metrics
                        ):
                            metrics = job_status.training_metrics
                            if "train_loss" in metrics:
                                status_msg += (
                                    f" | Train Loss: {metrics['train_loss']:.4f}"
                                )
                            if "eval_loss" in metrics:
                                status_msg += f" | Val Loss: {metrics['eval_loss']:.4f}"

                        logger.urgent(status_msg)

                    status_counter += 1

                    if job_status.status == "completed":
                        break
                    elif job_status.status == "failed":
                        raise RuntimeError(f"Fine-tuning failed: {job_status.error}")

                    await asyncio.sleep(30)

                # Together AI uses 'output_name' instead of 'fine_tuned_model'
                fine_tuned_model_name = job_status.output_name

            # Update the metadata with the actual fine-tuned model name
            updated_metadata = {
                "api_model_id": fine_tuned_model_name,
                "provider": self.model_provider,
                "num_samples": len(samples),
                "num_val_samples": len(validation_samples) if validation_samples else 0,
                "training_file": str(training_file),
                "validation_file": str(validation_file) if validation_file else None,
                **metadata,
            }

            # Update the temp_model with the actual fine-tuned model name
            temp_model.model_name = fine_tuned_model_name

            # Save updated metadata
            model_dir = Path("data/models") / temp_model.colloquial_name
            dump_file(
                model_dir / "metadata.json", updated_metadata, indent=2, default=str
            )

            logger.urgent(
                f"[{model_display_name}] Fine-tuning completed. Model: {fine_tuned_model_name}"
            )

            return temp_model

        finally:
            # Clean up temporary files
            if training_file.exists():
                training_file.unlink()
            if validation_file and validation_file.exists():
                validation_file.unlink()

    async def train_rl_async(
        self,
        samples: list[Problem],  # List of Problem objects
        grader: Union[dict, Callable],
        validation_samples: list[Problem] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "APIModel":
        """
        Perform reinforcement learning using OpenAI's RL API.

        :param samples: List of Problem objects for training
        :param grader: Either a dict with full grader spec (for model or python grader) or a callable for python grader
        :param validation_samples: Optional list of Problem objects for validation
        :param metadata: Training metadata
        :return: New RL-trained APIModel instance
        """
        logger.major(
            "Starting OpenAI RL training with {} problems and {} validation problems",
            len(samples),
            len(validation_samples or []),
        )

        # Only OpenAI supports RL fine-tuning currently
        if self.model_provider != "openai":
            raise NotImplementedError(
                f"RL training not supported for provider '{self.model_provider}'. "
                "Only 'openai' provider supports RL training currently."
            )

        # Track if WandB is available for syncing
        use_wandb = False
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            try:
                from wandb.integration.openai.fine_tuning import WandbLogger

                use_wandb = True
            except ImportError:
                logger.major(
                    "WandB OpenAI integration not found. Install with: pip install wandb"
                )
                use_wandb = False

        # Create a deep copy early to get the colloquial_name for logging
        timestamp = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(0, 1000000)}"
        )
        temp_model = self.deep_copy(
            suffix_type="rl", suffix_data=samples, metadata=metadata
        )
        model_display_name = (
            temp_model.colloquial_name[:30] + "..."
            if len(temp_model.colloquial_name) > 30
            else temp_model.colloquial_name
        )

        # Prepare training data in JSONL format for RL
        # RL training data should only include the prompt (user messages), NOT the assistant response
        # The assistant response will be generated during training and graded
        def convert_problem(problem: Problem) -> dict:
            return {
                **(problem.aux_info or {}),
                **dataclasses.asdict(problem),
                "messages": [
                    {"role": "user", "content": problem.question}
                ],  # User message only
            }

        training_data = []
        for problem in samples:
            training_data.append(convert_problem(problem))

        # Save training data to temporary file
        training_file = Path("data/tmp") / f"rl_training_data_{timestamp}.jsonl"
        training_file.parent.mkdir(parents=True, exist_ok=True)

        with open(training_file, "w", encoding="utf-8") as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.major(
            f"[{model_display_name}] Saved {len(training_data)} RL training examples to {training_file}"
        )

        # Prepare validation data if provided
        validation_file = None
        if validation_samples:
            validation_data = []
            for problem in validation_samples:
                validation_data.append(convert_problem(problem))

            validation_file = Path("data/tmp") / f"rl_validation_data_{timestamp}.jsonl"
            with open(validation_file, "w", encoding="utf-8") as f:
                for item in validation_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            logger.major(
                f"[{model_display_name}] Saved {len(validation_data)} RL validation examples to {validation_file}"
            )

        # Prepare grader configuration using the new grader system
        from core.grader.schema import Grader, create_grader_from_spec

        if grader is None:
            raise ValueError(
                "Grader is required for RL training. Provide either:\n"
                "1. A Grader instance\n"
                "2. A dict with grader specification\n"
                "3. A Python callable for custom grading"
            )

        # Convert to Grader instance if needed
        if isinstance(grader, Grader):
            grader_instance = grader
        elif isinstance(grader, (dict, Callable)):
            grader_instance = create_grader_from_spec(grader)
        else:
            raise ValueError(
                f"Invalid grader type: {type(grader)}. Must be a Grader instance, dict, or callable."
            )

        # Get OpenAI spec from the grader
        grader_spec = grader_instance.to_openai_spec()
        logger.major(
            f"[{model_display_name}] Using grader: {grader_spec.get('type', 'unknown')} type"
        )

        try:
            from openai import OpenAI

            client = OpenAI()

            # Upload training file
            with open(training_file, "rb") as f:
                file_response = client.files.create(file=f, purpose="fine-tune")

            logger.major(
                f"[{model_display_name}] Uploaded RL training file: {file_response.id}"
            )

            # Upload validation file if provided
            validation_file_id = None
            if validation_file:
                with open(validation_file, "rb") as f:
                    val_file_response = client.files.create(file=f, purpose="fine-tune")
                validation_file_id = val_file_response.id
                logger.major(
                    f"[{model_display_name}] Uploaded RL validation file: {validation_file_id}"
                )

            # Create RL fine-tuning job with grader
            # According to OpenAI docs, grader goes in method.reinforcement.grader
            job_params = {
                "training_file": file_response.id,
                "model": self.model_name,
                "suffix": f"rl-{metadata.get('characteristics', timestamp[:8])}",
                "method": {
                    "type": "reinforcement",
                    "reinforcement": {
                        "grader": grader_spec,
                        "hyperparameters": {
                            "reasoning_effort": "medium",  # For o4-mini and other reasoning models
                            "n_epochs": metadata.get("num_epochs", 1),
                        },
                    },
                },
            }

            if validation_file_id:
                job_params["validation_file"] = validation_file_id

            logger.major(
                f"[{model_display_name}] Creating RL training job with {grader_spec['type']} grader"
            )

            fine_tune_job = client.fine_tuning.jobs.create(**job_params)

            logger.urgent(
                f"[{model_display_name}] Created RL training job: {fine_tune_job.id}"
            )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Sync with WandB if available
                if use_wandb:
                    try:
                        from wandb.integration.openai.fine_tuning import WandbLogger

                        event_loop = asyncio.get_event_loop()
                        event_loop.run_in_executor(
                            executor, WandbLogger.sync, fine_tune_job.id
                        )
                        logger.urgent(
                            f"[{model_display_name}] WandB syncing enabled for RL job: {fine_tune_job.id}"
                        )
                    except Exception as e:
                        logger.urgent(f"Failed to sync with WandB: {e}")

                # Wait for RL training to complete with status updates
                status_counter = 0
                while True:
                    job_status = client.fine_tuning.jobs.retrieve(fine_tune_job.id)

                    # Print status update every 3rd iteration (reduced frequency)
                    if status_counter % 3 == 0:
                        status_msg = f"[{model_display_name}] RL training status: {job_status.status}"

                        # Add metrics if available from OpenAI's training events
                        if job_status.status == "running":
                            try:
                                # Get latest training metrics from events
                                events = list(
                                    client.fine_tuning.jobs.list_events(
                                        fine_tuning_job_id=fine_tune_job.id, limit=10
                                    )
                                )
                                for event in events:
                                    if hasattr(event, "data") and event.data:
                                        if "train_loss" in event.data:
                                            status_msg += f" | Train Loss: {event.data['train_loss']:.4f}"
                                        if "valid_loss" in event.data:
                                            status_msg += f" | Val Loss: {event.data['valid_loss']:.4f}"
                                        if "mean_reward" in event.data:
                                            status_msg += f" | Mean Reward: {event.data['mean_reward']:.4f}"
                                        break  # Use first event with metrics
                            except Exception:
                                pass  # Ignore errors getting events

                        logger.urgent(status_msg)

                    status_counter += 1

                    if job_status.status == "succeeded":
                        break
                    elif job_status.status == "failed":
                        raise RuntimeError(f"RL training failed: {job_status.error}")

                    await asyncio.sleep(30)

            # Get the fine-tuned model name
            fine_tuned_model_name = job_status.fine_tuned_model

            # Update the metadata with the actual fine-tuned model name
            updated_metadata = {
                "api_model_id": fine_tuned_model_name,
                "provider": self.model_provider,
                "training_method": "reinforcement_learning",
                "num_samples": len(samples),
                "num_val_samples": len(validation_samples) if validation_samples else 0,
                "training_file": str(training_file),
                "validation_file": str(validation_file) if validation_file else None,
                "grader_type": grader_spec.get("type", "unknown"),
                **metadata,
            }

            # Update the temp_model with the actual fine-tuned model name
            temp_model.model_name = fine_tuned_model_name

            # Save updated metadata
            model_dir = Path("data/models") / temp_model.colloquial_name
            dump_file(
                model_dir / "metadata.json", updated_metadata, indent=2, default=str
            )

            logger.urgent(
                f"[{model_display_name}] RL training completed. Model: {fine_tuned_model_name}"
            )

            return temp_model

        finally:
            # Clean up temporary files
            if training_file.exists():
                training_file.unlink()
            if validation_file and validation_file.exists():
                validation_file.unlink()
