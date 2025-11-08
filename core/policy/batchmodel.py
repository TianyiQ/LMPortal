"""
BatchAPIModel - A sophisticated batching API model that accumulates requests and submits them
as batches to provider batch APIs for maximum cost efficiency.

This model waits for incoming inference requests and batches them together, submitting
to batch APIs when either:
1. A timeout period expires (default 10 minutes)
2. The batch size limit for the provider is reached

Supports OpenAI and Anthropic batch APIs through safety_tooling.
"""

import asyncio
import logging
import os
import time
import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import utils.path_utils

try:
    from safetytooling.apis.batch_api import BatchInferenceAPI
except ImportError:
    print("WARNING: SafetyTooling API loading failed, BatchAPIModel will not work.")
    BatchInferenceAPI = None

try:
    from safetytooling.data_models import (
        ChatMessage,
        LLMResponse,
        MessageRole,
        Prompt,
    )
except ImportError:
    print("WARNING: SafetyTooling not found, BatchAPIModel will not work.")
    ChatMessage = None
    LLMResponse = None
    MessageRole = None
    Prompt = None

from core.policy.apimodel import APIModel

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """Represents a pending inference request waiting to be batched."""

    request_id: str
    history: list[dict[str, str]]
    kwargs: dict[str, Any]
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchJob:
    """Represents a batch job submitted to a provider."""

    job_id: str
    batch_id: str
    requests: list[PendingRequest]
    submitted_at: float
    provider: str
    model_name: str


class BatchAPIModel(APIModel):
    """
    A batching API model that accumulates requests and submits them as batches
    to provider batch APIs for cost efficiency.

    This model only supports async inference and batches requests automatically.
    Regular infer_single() calls will raise NotImplementedError.
    """

    # Provider batch size limits (conservative estimates)
    BATCH_SIZE_LIMITS = {
        "anthropic": 50000,  # Conservative from 100k observed
        "openai": 40000,  # Conservative from 50k observed
        "together": None,  # Not supported
        "google": None,  # Not supported
        "deepseek": None,  # Not supported
    }

    # Provider-specific default timeouts (in seconds)
    DEFAULT_TIMEOUTS = {
        "anthropic": 74400,  # 24 hours
        "openai": 74400,  # 24 hours
    }

    def __init__(
        self,
        model_name: str,
        colloquial_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_provider: str = "auto",
        batch_timeout: Optional[float] = None,
        max_batch_size: Optional[int] = None,
        log_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        few_shot_examples: list[dict[str, str]] = None,
    ):
        """
        Initialize the BatchAPIModel.

        :param model_name: The name of the model
        :param colloquial_name: Human-readable name for the model
        :param system_prompt: System prompt to prepend to conversations
        :param model_provider: API provider (auto-detected if "auto")
        :param batch_timeout: Timeout in seconds before submitting batch (default: 600s)
        :param max_batch_size: Maximum batch size (default: provider limit)
        :param log_dir: Directory for batch logs
        :param cache_dir: Directory for batch cache
        """
        super().__init__(model_name, colloquial_name, system_prompt, model_provider, few_shot_examples)

        # Determine the actual provider
        self._actual_provider = self._detect_provider()

        # Validate provider support
        if self._actual_provider not in self.BATCH_SIZE_LIMITS:
            raise NotImplementedError(
                f"Batch API not supported for provider '{self._actual_provider}'. "
                f"Supported providers: {list(self.BATCH_SIZE_LIMITS.keys())}"
            )

        if self.BATCH_SIZE_LIMITS[self._actual_provider] is None:
            raise NotImplementedError(
                f"Batch API not implemented for provider '{self._actual_provider}'. "
                f"Only OpenAI and Anthropic batch APIs are currently supported."
            )

        # Set batch parameters
        self.batch_timeout = batch_timeout or self.DEFAULT_TIMEOUTS.get(self._actual_provider, 74400)  # 24 hours
        self.max_batch_size = max_batch_size or self.BATCH_SIZE_LIMITS[self._actual_provider]

        # Initialize batch API
        self.batch_api = BatchInferenceAPI(log_dir=log_dir or "default", cache_dir=cache_dir or "default")

        # Synchronization primitives
        self._pending_requests: list[PendingRequest] = []
        self._pending_lock = asyncio.Lock()
        self._batch_condition = asyncio.Condition(self._pending_lock)
        self._active_batches: dict[str, BatchJob] = {}
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._total_requests = 0
        self._total_batches = 0
        self._total_cost_savings = 0.0

        # Note: Batch processor will be started lazily when first request comes in
        logger.info(
            f"Initialized BatchAPIModel for {self._actual_provider} with "
            f"timeout={self.batch_timeout}s, max_batch_size={self.max_batch_size}"
        )

    def _detect_provider(self) -> str:
        """Detect the actual provider based on model name and provider setting."""
        if self.model_provider != "auto":
            return self.model_provider

        model_name_lower = self.model_name.lower()

        if any(name in model_name_lower for name in ["gpt", "o1", "openai"]):
            return "openai"
        elif any(name in model_name_lower for name in ["claude", "anthropic"]):
            return "anthropic"
        elif any(name in model_name_lower for name in ["gemini", "google"]):
            return "google"
        elif "together" in model_name_lower:
            return "together"
        elif "deepseek" in model_name_lower:
            return "deepseek"
        else:
            # Default fallback
            return "openai"

    def _start_batch_processor(self):
        """Start the background batch processor task."""
        if self._batch_processor_task is None or self._batch_processor_task.done():
            try:
                self._batch_processor_task = asyncio.create_task(self._batch_processor())
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    # This is expected during initialization - task will be started when needed
                    logger.debug("No event loop available for batch processor - will start later")
                else:
                    raise

    async def _batch_processor(self):
        """
        Background task that processes batches based on timeout or size limits.

        This coroutine runs continuously and:
        1. Waits for requests to accumulate
        2. Submits batches when timeout expires or size limit reached
        3. Monitors submitted batches for completion
        """
        logger.info("Batch processor started")

        try:
            while not self._shutdown_event.is_set():
                async with self._batch_condition:
                    # Wait for requests or timeout
                    try:
                        await asyncio.wait_for(
                            self._batch_condition.wait_for(
                                lambda: len(self._pending_requests) > 0 or self._shutdown_event.is_set()
                            ),
                            timeout=self.batch_timeout,
                        )
                    except TimeoutError:
                        pass  # Timeout is expected - check if we have requests to process

                    if self._shutdown_event.is_set():
                        break

                    # Check if we should submit a batch
                    should_submit = len(self._pending_requests) >= self.max_batch_size or (  # Size limit reached
                        len(self._pending_requests) > 0 and self._should_submit_by_timeout()
                    )  # Timeout reached

                    if should_submit:
                        await self._submit_current_batch()

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Batch processor error: {e}")
        finally:
            logger.info("Batch processor stopped")

    def _should_submit_by_timeout(self) -> bool:
        """Check if any pending request has exceeded the timeout."""
        if not self._pending_requests:
            return False

        current_time = time.time()
        oldest_request = min(self._pending_requests, key=lambda r: r.created_at)
        return (current_time - oldest_request.created_at) >= self.batch_timeout

    async def _submit_current_batch(self):
        """Submit the current batch of pending requests."""
        if not self._pending_requests:
            return

        # Extract requests to batch (make a copy)
        batch_requests = self._pending_requests.copy()
        self._pending_requests.clear()

        batch_id = f"batch_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        logger.info(f"Submitting batch {batch_id} with {len(batch_requests)} requests")

        try:
            # Prepare prompts for batch API
            prompts = []
            for req in batch_requests:
                # Convert history to Prompt format
                if isinstance(req.history, str):
                    history = [{"role": "user", "content": req.history}]
                else:
                    history = deepcopy(req.history)

                # Add system prompt if needed
                if self.system_prompt and not any(msg.get("role") == "system" for msg in history):
                    history.insert(0, {"role": "system", "content": self.system_prompt})

                prompt = Prompt(
                    messages=[ChatMessage(content=msg["content"], role=MessageRole(msg["role"])) for msg in history]
                )
                prompts.append(prompt)

            # Prepare batch kwargs
            batch_kwargs = {
                "model_id": self.model_name,
                "prompts": prompts,
            }

            # Add common parameters from first request (they should be similar)
            if batch_requests:
                first_kwargs = batch_requests[0].kwargs
                if "temperature" in first_kwargs:
                    batch_kwargs["temperature"] = first_kwargs["temperature"]
                if "max_tokens" in first_kwargs:
                    batch_kwargs["max_tokens"] = first_kwargs["max_tokens"]
                else:
                    # Set default max_tokens for Anthropic (required)
                    if self._actual_provider == "anthropic":
                        batch_kwargs["max_tokens"] = 200

            # Submit batch
            batch_job = BatchJob(
                job_id=batch_id,
                batch_id="",  # Will be set by API
                requests=batch_requests,
                submitted_at=time.time(),
                provider=self._actual_provider,
                model_name=self.model_name,
            )

            # Submit to batch API in background
            asyncio.create_task(self._process_batch(batch_job, batch_kwargs))

            self._total_batches += 1

        except Exception as e:
            logger.error(f"Failed to submit batch {batch_id}: {e}")
            # Fail all requests in the batch
            for req in batch_requests:
                if not req.future.done():
                    req.future.set_exception(e)

    async def _process_batch(self, batch_job: BatchJob, batch_kwargs: dict[str, Any]):
        """Process a batch job asynchronously with retry logic."""
        max_retries = 3
        retry_delay = 60  # Start with 1 minute delay

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Processing batch {batch_job.job_id} with {len(batch_job.requests)} requests (attempt {attempt + 1}/{max_retries})"
                )

                # Submit to batch API and wait for completion
                responses, actual_batch_id = await self.batch_api(
                    model_id=batch_kwargs["model_id"],
                    prompts=batch_kwargs["prompts"],
                    max_tokens=batch_kwargs.get("max_tokens"),
                    temperature=batch_kwargs.get("temperature"),
                    use_cache=False,  # Don't use cache for batch requests to avoid conflicts
                    **{
                        k: v
                        for k, v in batch_kwargs.items()
                        if k not in ["model_id", "prompts", "max_tokens", "temperature"]
                    },
                )
                batch_job.batch_id = actual_batch_id

                logger.info(f"Batch {batch_job.job_id} completed with batch_id {actual_batch_id}")

                # Distribute results to waiting requests
                await self._distribute_batch_results(batch_job, responses)
                return  # Success, exit retry loop

            except Exception as e:
                logger.error(f"Batch {batch_job.job_id} failed on attempt {attempt + 1}: {e}")

                if attempt < max_retries - 1:
                    # Wait before retrying with exponential backoff
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    # Final attempt failed, fail all requests
                    logger.error(f"Batch {batch_job.job_id} failed after {max_retries} attempts")
                    for req in batch_job.requests:
                        if not req.future.done():
                            req.future.set_exception(e)

    async def _distribute_batch_results(self, batch_job: BatchJob, responses: list[LLMResponse]):
        """Distribute batch results to the corresponding request futures."""
        # Safety_tooling returns responses in the same order as input prompts
        if len(responses) != len(batch_job.requests):
            logger.error(
                f"Response count mismatch: got {len(responses)} responses for {len(batch_job.requests)} requests"
            )
            # Fail all requests if we can't match them properly
            for req in batch_job.requests:
                if not req.future.done():
                    req.future.set_exception(
                        RuntimeError(
                            f"Response count mismatch: got {len(responses)} responses for {len(batch_job.requests)} requests"
                        )
                    )
            return

        # Distribute results by index
        for i, (req, response) in enumerate(zip(batch_job.requests, responses, strict=False)):
            try:
                if response is not None and response.completion:
                    req.future.set_result(response.completion)
                elif response is None:
                    req.future.set_exception(RuntimeError(f"Request {req.request_id} failed: batch API returned None"))
                else:
                    req.future.set_exception(RuntimeError(f"Empty response for request {req.request_id}"))
            except Exception as e:
                logger.error(f"Error distributing result for request {req.request_id}: {e}")
                if not req.future.done():
                    req.future.set_exception(e)

    async def infer_single_async(
        self,
        history: Union[list[dict[str, str]], str],
        disable_system_prompt: bool = False,
        disable_logging: bool = True,
        **kwargs,
    ) -> str:
        """
        Queue a single inference request for batching.

        This method adds the request to the pending queue and waits for the
        batch to be processed and results returned.
        """
        history = self._prepend_few_shot_to_history(history)

        # Ensure batch processor is running
        self._start_batch_processor()

        request_id = uuid.uuid4().hex
        future = asyncio.Future()

        # Process history
        if isinstance(history, str):
            processed_history = [{"role": "user", "content": history}]
        else:
            processed_history = deepcopy(history)

        # Handle system prompt
        if not disable_system_prompt and self.system_prompt:
            if not any(msg.get("role") == "system" for msg in processed_history):
                processed_history.insert(0, {"role": "system", "content": self.system_prompt})

        # Set default parameters
        if "temperature" not in kwargs:
            kwargs["temperature"] = self.temperature
        if "presence_penalty" not in kwargs:
            kwargs["presence_penalty"] = self.presence_penalty

        # Remove presence_penalty for Claude models
        if "claude" in self.model_name and "presence_penalty" in kwargs:
            if float(kwargs["presence_penalty"]) != 0:
                warnings.warn("Claude API doesn't support presence penalty. Ignoring the penalty.")
            del kwargs["presence_penalty"]

        # Create pending request
        pending_request = PendingRequest(request_id=request_id, history=processed_history, kwargs=kwargs, future=future)

        # Add to pending requests
        async with self._batch_condition:
            self._pending_requests.append(pending_request)
            self._total_requests += 1
            self._batch_condition.notify()

        logger.debug(f"Queued request {request_id}, {len(self._pending_requests)} pending")

        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise

    async def infer_batch_async(self, histories: list[Union[list[dict[str, str]], str]], **kwargs) -> list[str]:
        """
        Process multiple inference requests concurrently.

        This method submits all requests concurrently and waits for all
        to complete. They may be batched together or spread across multiple batches.
        """
        tasks = []
        for history in histories:
            task = asyncio.create_task(self.infer_single_async(history, **kwargs))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get batching statistics."""
        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "pending_requests": len(self._pending_requests),
            "active_batches": len(self._active_batches),
            "provider": self._actual_provider,
            "batch_timeout": self.batch_timeout,
            "max_batch_size": self.max_batch_size,
            "average_batch_size": self._total_requests / max(self._total_batches, 1),
        }

    async def logprobs_single_async(
        self, dialogue: list[dict[str, str]], return_summed: bool = True
    ) -> float | list[float]:
        """Calculate sum of log probabilities for a single dialogue."""
        raise NotImplementedError("BatchAPIModel does not support logprob calculation.")

    async def flush_pending(self, timeout: float = 30.0) -> int:
        """
        Force submission of all pending requests and wait for completion.

        :param timeout: Maximum time to wait for completion
        :return: Number of requests that were flushed
        """
        async with self._batch_condition:
            pending_count = len(self._pending_requests)
            if pending_count > 0:
                logger.info(f"Flushing {pending_count} pending requests")
                await self._submit_current_batch()
                self._batch_condition.notify()

        # Wait for all pending requests to complete
        start_time = time.time()
        while time.time() - start_time < timeout:
            async with self._pending_lock:
                if len(self._pending_requests) == 0:
                    break
            await asyncio.sleep(0.1)

        return pending_count

    async def shutdown(self):
        """Shutdown the batch processor and complete pending requests."""
        logger.info("Shutting down BatchAPIModel")

        # Flush any pending requests
        await self.flush_pending(timeout=60.0)

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for batch processor to finish
        if self._batch_processor_task and not self._batch_processor_task.done():
            try:
                await asyncio.wait_for(self._batch_processor_task, timeout=10.0)
            except TimeoutError:
                logger.warning("Batch processor did not shutdown gracefully")
                self._batch_processor_task.cancel()

        logger.info("BatchAPIModel shutdown complete")

    def __del__(self):
        """Cleanup on deletion."""
        if (
            hasattr(self, "_batch_processor_task")
            and self._batch_processor_task
            and not self._batch_processor_task.done()
        ):
            self._batch_processor_task.cancel()
