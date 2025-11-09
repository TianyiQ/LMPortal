"""
Ray-based API model implementation for maximum throughput on multicore servers.
Optimized for high-concurrency API calling with distributed load balancing.
"""

import asyncio
import os
import random
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Optional

import ray

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
    print("WARNING: SafetyTooling not found, RayModel will not work.")
    ChatMessage = None
    LLMResponse = None
    MessageRole = None
    Prompt = None
    StopReason = None


from core.policy.apimodel import APIModel
from utils.io_utils import dump_file

# Import direct API clients for reduced overhead
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None

try:
    from together import AsyncTogether

    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    AsyncTogether = None

try:
    import httpx

    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False
    httpx = None

try:
    import vertexai
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    api_key = os.getenv("GOOGLE_API_KEY")
    vertexai.init(project=project_id, location=location, api_key=api_key)
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False
    vertexai = None
    TextEmbeddingModel = None


@dataclass
class APIRequest:
    """Request structure for Ray workers."""

    prompt: Prompt
    kwargs: dict[str, Any]
    request_id: str
    model_name: str
    model_provider: str
    max_concurrent_per_worker: int
    max_retries: int = 3
    retry_delay: float = 1.0
    api_timeout: float = 600.0


@dataclass
class APIResponse:
    """Response structure from Ray workers."""

    completion: str
    request_id: str
    worker_id: str
    processing_time: float
    error: Optional[str] = None
    retry_count: int = 0
    final_attempt: bool = True


@dataclass
class LogprobRequest:
    """Request structure for logprob calculations."""

    dialogue: list[dict[str, str]]
    return_summed: bool
    request_id: str
    model_name: str
    model_provider: str
    system_prompt: Optional[str]
    max_concurrent_per_worker: int
    max_retries: int = 3
    retry_delay: float = 1.0
    kwargs: Optional[dict[str, Any]] = None  # Additional kwargs like debug
    api_timeout: float = 600.0


@dataclass
class LogprobResponse:
    """Response structure for logprob calculations."""

    logprobs: float | list[float]  # Sum or list of per-token logprobs
    request_id: str
    worker_id: str
    processing_time: float
    error: Optional[str] = None
    retry_count: int = 0


@ray.remote
class APIWorker:
    """Ray actor for distributed API calling with shared worker pool."""

    def __init__(
        self,
        worker_id: str,
        model_name: str = "shared",
        model_provider: str = "auto",
        max_concurrent_requests: int = 500,
    ):
        self.worker_id = worker_id
        self.max_concurrent_requests = max_concurrent_requests

        # Each worker gets its own API instance for connection pooling
        self.api = (
            InferenceAPI(  # type: ignore
                together_num_threads=max_concurrent_requests,
                anthropic_num_threads=max_concurrent_requests,
                openai_num_threads=max_concurrent_requests,
                openai_s2s_num_threads=max_concurrent_requests,
                gpt4o_s2s_rpm_cap=50000,  # Higher RPM per worker
                no_cache=True,
                prompt_history_dir=None,
            )
            if InferenceAPI is not None
            else None
        )

        # Per-instance semaphore tracking
        self.semaphores = {}  # Will store semaphores per instance

        # Performance tracking
        self.requests_processed = 0
        self.total_processing_time = 0.0
        self.retry_stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_after_retries": 0,
            "non_retryable_errors": 0,
        }

        # Initialize direct API clients for reduced overhead
        self.openai_client = None
        self.anthropic_client = None
        self.together_client = None
        self.openrouter_client = None

        if OPENAI_AVAILABLE:
            try:
                self.openai_client = openai.AsyncClient()
            except Exception as e:
                print(f"WARNING: Could not initialize OpenAI client: {e}")

        if ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = AsyncAnthropic()
            except Exception as e:
                print(f"WARNING: Could not initialize Anthropic client: {e}")

        if TOGETHER_AVAILABLE:
            try:
                self.together_client = AsyncTogether()
            except Exception as e:
                print(f"WARNING: Could not initialize TogetherAI client: {e}")

        if OPENROUTER_AVAILABLE:
            try:
                # OpenRouter uses HTTP client with Bearer token authentication
                self.openrouter_client = httpx.AsyncClient(
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", ""),
                        "X-Title": os.getenv("OPENROUTER_TITLE", "Ray API Model"),
                    },
                )
            except Exception as e:
                print(f"WARNING: Could not initialize OpenRouter client: {e}")
                self.openrouter_client = None

    def _is_gpt_model(self, model_name: str) -> bool:
        """Check if the model is a GPT model that should use direct OpenAI API."""
        model_lower = model_name.lower()
        return any(
            gpt_indicator in model_lower
            for gpt_indicator in ["gpt-", "o1-", "o3-", "ft:gpt"]
        )

    def _is_anthropic_model(self, model_name: str) -> bool:
        """Check if the model is an Anthropic model that should use direct Anthropic API."""
        model_lower = model_name.lower()
        return model_lower.startswith("claude") or any(
            indicator in model_lower
            for indicator in [
                "claude-3-5-sonnet",
                "claude-3-5-haiku",
                "claude-3-opus",
                "research-claude",
            ]
        )

    def _is_together_model(self, model_name: str) -> bool:
        """Check if the model is a TogetherAI model that should use direct TogetherAI API."""
        model_lower = model_name.lower()
        return any(
            indicator in model_lower
            for indicator in [
                "llama",
                "deepseek",
                "qwen",
                "mixtral",
                "mistral",
                "scalesafetyresearch",
                "together",
                "gemma",
            ]
        )

    def _is_openrouter_model(self, model_provider: str) -> bool:
        """Check if the model should use OpenRouter API based on provider."""
        return "openrouter" in model_provider

    async def _call_openai_directly(
        self, request: APIRequest, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Make a direct OpenAI API call, bypassing safety_tooling."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not available")

        start_time = time.time()
        api_start = time.time()

        # Convert Prompt to OpenAI format
        messages = []
        for msg in request.prompt.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        if "o3" in request.model_name or "o4" in request.model_name:
            del kwargs["temperature"]

        if "seed" not in kwargs:
            kwargs["seed"] = random.randint(0, 2**30 - 1)

        try:
            # Make the API call with timeout
            api_response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    messages=messages, model=request.model_name, **kwargs
                ),
                timeout=request.api_timeout,
            )
        except TimeoutError as timeout_err:
            raise RuntimeError(
                f"OpenAI API call timed out after {request.api_timeout} seconds"
            ) from timeout_err

        api_duration = time.time() - api_start
        duration = time.time() - start_time

        # Convert response to dictionary format
        choice = api_response.choices[0]

        # Calculate cost (simplified - could import from safety_tooling if needed)
        cost = 0.0
        if hasattr(api_response, "usage") and api_response.usage:
            # Rough cost estimation - this could be made more accurate
            prompt_tokens = api_response.usage.prompt_tokens
            completion_tokens = api_response.usage.completion_tokens
            # Simplified cost calculation (adjust rates as needed)
            cost = (prompt_tokens * 0.00001) + (completion_tokens * 0.00003)

        return {
            "model_id": request.model_name,
            "completion": choice.message.content or "",
            "stop_reason": choice.finish_reason,
            "cost": cost,
            "duration": duration,
            "api_duration": api_duration,
        }

    async def _call_anthropic_directly(
        self, request: APIRequest, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Make a direct Anthropic API call, bypassing safety_tooling."""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not available")

        start_time = time.time()
        api_start = time.time()

        # Convert Prompt to Anthropic format (separate system prompt from messages)
        messages = []
        system_prompt = None

        for msg in request.prompt.messages:
            if msg.role.value == "system":
                system_prompt = msg.content
            else:
                messages.append({"role": msg.role.value, "content": msg.content})

        # Prepare kwargs for Anthropic API
        anthropic_kwargs = kwargs.copy()
        anthropic_kwargs.setdefault("max_tokens", 8192)  # Required for Anthropic

        # Add system prompt if present
        if system_prompt:
            anthropic_kwargs["system"] = system_prompt

        try:
            # Make the API call with timeout
            api_response = await asyncio.wait_for(
                self.anthropic_client.messages.create(
                    messages=messages, model=request.model_name, **anthropic_kwargs
                ),
                timeout=request.api_timeout,
            )
        except TimeoutError as timeout_err:
            raise RuntimeError(
                f"Anthropic API call timed out after {request.api_timeout} seconds"
            ) from timeout_err

        api_duration = time.time() - api_start
        duration = time.time() - start_time

        # Extract content from response
        content = ""
        reasoning_content = None

        if api_response.content:
            # Handle reasoning models (thinking content)
            if (
                hasattr(api_response.content[0], "type")
                and api_response.content[0].type == "thinking"
            ):
                reasoning_content = api_response.content[0].content
                if len(api_response.content) > 1:
                    content = api_response.content[1].text
            else:
                content = (
                    api_response.content[0].text if api_response.content[0].text else ""
                )

        # Calculate cost (simplified)
        cost = 0.0
        if hasattr(api_response, "usage") and api_response.usage:
            # Simplified cost calculation for Anthropic (adjust rates as needed)
            input_tokens = api_response.usage.input_tokens
            output_tokens = api_response.usage.output_tokens
            cost = (input_tokens * 0.000003) + (
                output_tokens * 0.000015
            )  # Claude-3.5 Sonnet rates

        return {
            "model_id": request.model_name,
            "completion": content,
            "stop_reason": api_response.stop_reason,
            "cost": cost,
            "duration": duration,
            "api_duration": api_duration,
            "reasoning_content": reasoning_content,
        }

    async def _call_together_directly(
        self, request: APIRequest, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Make a direct TogetherAI API call, bypassing safety_tooling."""
        if not self.together_client:
            raise RuntimeError("TogetherAI client not available")

        start_time = time.time()
        api_start = time.time()

        # Convert Prompt to OpenAI format (TogetherAI uses OpenAI-compatible format)
        messages = []
        for msg in request.prompt.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Handle logprobs parameter conversion for TogetherAI
        together_kwargs = kwargs.copy()
        if "logprobs" in together_kwargs:
            logprobs_value = together_kwargs.pop("logprobs")
            if isinstance(logprobs_value, int) and logprobs_value > 0:
                together_kwargs["top_logprobs"] = logprobs_value
                together_kwargs["logprobs"] = True

        if "seed" not in together_kwargs:
            together_kwargs["seed"] = random.randint(0, 2**30 - 1)

        try:
            # Make the API call with timeout
            api_response = await asyncio.wait_for(
                self.together_client.chat.completions.create(
                    messages=messages, model=request.model_name, **together_kwargs
                ),
                timeout=request.api_timeout,
            )
        except TimeoutError as timeout_err:
            raise RuntimeError(
                f"TogetherAI API call timed out after {request.api_timeout} seconds"
            ) from timeout_err

        api_duration = time.time() - api_start
        duration = time.time() - start_time

        # Convert response
        choice = api_response.choices[0]

        # Calculate cost (simplified - TogetherAI often doesn't provide usage data)
        cost = 0.0
        if hasattr(api_response, "usage") and api_response.usage:
            prompt_tokens = api_response.usage.prompt_tokens
            completion_tokens = api_response.usage.completion_tokens
            # Simplified cost calculation (adjust rates as needed)
            cost = (prompt_tokens * 0.0000002) + (
                completion_tokens * 0.0000006
            )  # Rough TogetherAI rates

        return {
            "model_id": request.model_name,
            "completion": choice.message.content or "",
            "stop_reason": choice.finish_reason or "unspecified",
            "cost": cost,
            "duration": duration,
            "api_duration": api_duration,
        }

    async def _call_together_completion_directly(
        self, request: LogprobRequest
    ) -> dict[str, Any]:
        """Make a direct Together completion API call with echo for logprobs."""
        import json
        from copy import deepcopy

        import aiohttp

        # Get Together API key
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY not found in environment")

        start_time = time.time()
        api_start = time.time()

        # Convert dialogue to completion format
        prompt_parts = []

        # Add system prompt if configured
        history = deepcopy(request.dialogue)
        if request.system_prompt and (
            not history or history[0].get("role") != "system"
        ):
            history.insert(0, {"role": "system", "content": request.system_prompt})

        # Format conversation as a completion prompt
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

        # If the last message was from user, add "Assistant:" to trigger completion
        if history and history[-1]["role"] == "user":
            prompt += "Assistant:"

        # Prepare the API request
        url = "https://api.together.xyz/v1/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Request parameters for Together completion API
        if history and history[-1]["role"] == "assistant":
            max_new_tokens = 1  # Minimal generation
        else:
            max_new_tokens = 10  # Allow small generation after user message

        data = {
            "model": request.model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,  # Deterministic
            "echo": True,  # Critical: return prompt + completion with logprobs
            "logprobs": 1,  # Return top 1 logprob per token
            "stop": ["<|endoftext|>", "<|eot_id|>", "\n\n", "User:", "Assistant:"],
        }

        # Make the API request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=request.api_timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Together API error (status {response.status}): {error_text}"
                        )

                    result = await response.json()

                    # Debug output if requested
                    if request.kwargs and request.kwargs.get("debug", False):
                        print("DEBUG: Together Completion API Response:")
                        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])

                    api_duration = time.time() - api_start
                    duration = time.time() - start_time

                    # Extract logprobs from the response
                    all_token_logprobs = []

                    # Get prompt logprobs (echoed input)
                    if "prompt" in result and result["prompt"]:
                        prompt_data = result["prompt"][0]
                        if "logprobs" in prompt_data:
                            prompt_logprobs = prompt_data["logprobs"]
                            if "token_logprobs" in prompt_logprobs:
                                all_token_logprobs.extend(
                                    prompt_logprobs["token_logprobs"]
                                )

                    # Get any generated token logprobs
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        if "logprobs" in choice:
                            choice_logprobs = choice["logprobs"]
                            if "token_logprobs" in choice_logprobs:
                                all_token_logprobs.extend(
                                    choice_logprobs["token_logprobs"]
                                )

                    if not all_token_logprobs:
                        raise ValueError(
                            "No token_logprobs found in Together API response"
                        )

                    # Filter out None values
                    valid_logprobs = [lp for lp in all_token_logprobs if lp is not None]

                    if not valid_logprobs:
                        raise ValueError("No valid logprobs found (all were None)")

                    # Return based on return_summed parameter
                    if request.return_summed:
                        logprobs_result = float(sum(valid_logprobs))
                    else:
                        logprobs_result = [float(lp) for lp in valid_logprobs]

                    return {
                        "logprobs": logprobs_result,
                        "duration": duration,
                        "api_duration": api_duration,
                    }

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error calling Together API: {e}") from e
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse Together API response: {e}") from e

    async def _call_openrouter_directly(
        self, request: APIRequest, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Make a direct OpenRouter API call."""
        if not self.openrouter_client:
            raise RuntimeError("OpenRouter client not available")

        start_time = time.time()
        api_start = time.time()

        # Convert Prompt to OpenRouter format (uses OpenAI-compatible format)
        messages = []
        for msg in request.prompt.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Prepare request payload
        payload = {"model": request.model_name, "messages": messages, **kwargs}

        # Remove OpenRouter-incompatible parameters
        openrouter_kwargs = payload.copy()
        openrouter_kwargs.pop("force_provider", None)
        openrouter_kwargs.pop("max_attempts_per_api_call", None)

        if "seed" not in openrouter_kwargs:
            openrouter_kwargs["seed"] = random.randint(0, 2**30 - 1)

        # Debug logging
        if int(os.getenv("DEBUG", "0")) >= 2:
            print(f"OpenRouter request payload: {openrouter_kwargs}")

        try:
            # Make the API call with timeout
            response = await self.openrouter_client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=openrouter_kwargs,
                timeout=request.api_timeout,
            )
            response.raise_for_status()
            api_response = response.json()

            # Debug logging
            if int(os.getenv("DEBUG", "0")) >= 2:
                print(f"OpenRouter response: {api_response}")

        except httpx.TimeoutException as timeout_err:
            raise RuntimeError(
                f"OpenRouter API call timed out after {request.api_timeout} seconds"
            ) from timeout_err
        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            if int(os.getenv("DEBUG", "0")):
                print(f"OpenRouter HTTP error {e.response.status_code}: {error_text}")
            raise RuntimeError(
                f"OpenRouter API call failed with status {e.response.status_code}: {error_text}"
            ) from e
        except Exception as e:
            if int(os.getenv("DEBUG", "0")):
                print(f"OpenRouter API exception: {e}")
            raise RuntimeError(f"OpenRouter API call failed: {e}") from e

        api_duration = time.time() - api_start
        duration = time.time() - start_time

        # Extract response content
        if not api_response.get("choices"):
            # Provide more detailed error information
            error_details = (
                f"No choices in OpenRouter API response. Response: {api_response}"
            )
            if "error" in api_response:
                error_info = api_response["error"]
                if isinstance(error_info, dict):
                    error_msg = error_info.get("message", "Unknown error")
                    error_type = error_info.get("type", "Unknown type")
                    error_code = error_info.get("code", "Unknown code")
                    error_details = f"OpenRouter API error: {error_msg} (type: {error_type}, code: {error_code})"
                else:
                    error_details = f"OpenRouter API error: {error_info}"
            raise RuntimeError(error_details)

        choice = api_response["choices"][0]
        content = choice.get("message", {}).get("content", "")

        # Calculate cost using OpenRouter's usage data
        cost = 0.0
        if "usage" in api_response:
            usage = api_response["usage"]
            # OpenRouter provides cost information when available
            if "total_cost" in usage:
                cost = float(usage["total_cost"])
            elif "prompt_tokens" in usage and "completion_tokens" in usage:
                # Fallback to rough estimation if exact cost not provided
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
                # Use generic rates as fallback (OpenRouter rates vary by model)
                cost = (prompt_tokens * 0.000001) + (completion_tokens * 0.000003)

        return {
            "model_id": request.model_name,
            "completion": content,
            "stop_reason": choice.get("finish_reason", "stop"),
            "cost": cost,
            "duration": duration,
            "api_duration": api_duration,
        }

    async def process_request(self, request: APIRequest) -> APIResponse:
        """Process a single API request with per-instance rate limiting and retry mechanism."""
        return await self._process_request_with_retry(request, 0)

    async def _process_request_with_retry(
        self, request: APIRequest, retry_count: int
    ) -> APIResponse:
        """Internal method to handle retries."""
        start_time = time.time()

        if int(os.getenv("DEBUG", "0")) >= 2:  # radical debug mode
            print(
                f"Processing {request.model_name} request {request.request_id}, retried {retry_count} times, max_retries {request.max_retries}, retry_delay {request.retry_delay}"
            )

        if int(os.getenv("NO_RETRY", "0")):
            request.max_retries = 0
            request.retry_delay = 0
            retry_count = 0
            request.kwargs["max_attempts_per_api_call"] = 1

        # Get or create semaphore for this specific instance's concurrency limit
        instance_key = f"{request.model_name}_{request.max_concurrent_per_worker}"
        if instance_key not in self.semaphores:
            self.semaphores[instance_key] = asyncio.Semaphore(
                request.max_concurrent_per_worker
            )

        instance_semaphore = self.semaphores[instance_key]

        async with instance_semaphore:

            async def _call_api():
                kwargs = request.kwargs.copy()
                if request.model_provider != "auto":
                    kwargs["force_provider"] = request.model_provider

                # Use direct APIs for major providers to reduce overhead
                # Check OpenRouter first when explicitly specified
                if (
                    self._is_openrouter_model(request.model_provider)
                    and self.openrouter_client
                ):
                    if (int(os.getenv("DEBUG", "0")) and random.random() < 0.01) or int(
                        os.getenv("DEBUG", "0")
                    ) >= 2:  # 1% debug rate
                        print(
                            f"Ray worker {self.worker_id}: Using direct OpenRouter API for {request.model_name}"
                        )

                    # Remove safety_tooling specific kwargs that OpenRouter doesn't understand
                    direct_kwargs = kwargs.copy()
                    direct_kwargs.pop("force_provider", None)
                    direct_kwargs.pop("max_attempts_per_api_call", None)

                    direct_response = await self._call_openrouter_directly(
                        request, direct_kwargs
                    )
                    # Wrap in list to match safety_tooling format
                    response = [direct_response]

                elif self._is_gpt_model(request.model_name) and self.openai_client:
                    if (int(os.getenv("DEBUG", "0")) and random.random() < 0.01) or int(
                        os.getenv("DEBUG", "0")
                    ) >= 2:  # 1% debug rate
                        print(
                            f"Ray worker {self.worker_id}: Using direct OpenAI API for {request.model_name}"
                        )

                    # Remove safety_tooling specific kwargs that OpenAI doesn't understand
                    direct_kwargs = kwargs.copy()
                    direct_kwargs.pop("force_provider", None)
                    direct_kwargs.pop("max_attempts_per_api_call", None)

                    direct_response = await self._call_openai_directly(
                        request, direct_kwargs
                    )
                    # Wrap in list to match safety_tooling format
                    response = [direct_response]

                elif (
                    self._is_anthropic_model(request.model_name)
                    and self.anthropic_client
                ):
                    if (int(os.getenv("DEBUG", "0")) and random.random() < 0.01) or int(
                        os.getenv("DEBUG", "0")
                    ) >= 2:  # 1% debug rate
                        print(
                            f"Ray worker {self.worker_id}: Using direct Anthropic API for {request.model_name}"
                        )

                    # Remove safety_tooling specific kwargs that Anthropic doesn't understand
                    direct_kwargs = kwargs.copy()
                    direct_kwargs.pop("force_provider", None)
                    direct_kwargs.pop("max_attempts_per_api_call", None)

                    direct_response = await self._call_anthropic_directly(
                        request, direct_kwargs
                    )
                    # Wrap in list to match safety_tooling format
                    response = [direct_response]

                elif (
                    self._is_together_model(request.model_name) and self.together_client
                ):
                    if (int(os.getenv("DEBUG", "0")) and random.random() < 0.01) or int(
                        os.getenv("DEBUG", "0")
                    ) >= 2:  # 1% debug rate
                        print(
                            f"Ray worker {self.worker_id}: Using direct TogetherAI API for {request.model_name}"
                        )

                    # Remove safety_tooling specific kwargs that TogetherAI doesn't understand
                    direct_kwargs = kwargs.copy()
                    direct_kwargs.pop("force_provider", None)
                    direct_kwargs.pop("max_attempts_per_api_call", None)

                    direct_response = await self._call_together_directly(
                        request, direct_kwargs
                    )
                    # Wrap in list to match safety_tooling format
                    response = [direct_response]

                else:
                    if (int(os.getenv("DEBUG", "0")) and random.random() < 0.01) or int(
                        os.getenv("DEBUG", "0")
                    ) >= 2:  # 1% debug rate
                        print(
                            f"Ray worker {self.worker_id}: Using safety_tooling for {request.model_name}"
                        )

                    if "o3" in request.model_name or "o4" in request.model_name:
                        del kwargs["temperature"]

                    if "seed" not in kwargs:
                        kwargs["seed"] = random.randint(0, 2**30 - 1)

                    # Use safety_tooling for other models (Google, etc.)
                    response: list[LLMResponse] = await self.api(
                        model_id=request.model_name,
                        prompt=request.prompt,
                        print_prompt_and_response=False,
                        **kwargs,
                    )

                try:
                    # Handle both dictionary and LLMResponse formats
                    if isinstance(response[0], dict):
                        completion = response[0]["completion"]
                    else:
                        completion = response[0].completion
                except Exception:
                    # Handle nested list format
                    if isinstance(response[0][0], dict):
                        completion = response[0][0]["completion"]
                    else:
                        completion = response[0][0].completion

                processing_time = time.time() - start_time
                self.requests_processed += 1
                self.total_processing_time += processing_time

                if (int(os.getenv("DEBUG", "0")) and random.random() < 0.1) or int(
                    os.getenv("DEBUG", "0")
                ) >= 2:
                    print(
                        f"Completed {request.model_name} request {request.request_id} of length: {sum(len(msg.content) for msg in request.prompt.messages)} in {len(request.prompt.messages)} turns. Response length: {len(completion)}"
                    )
                    print(f"Ray worker {self.worker_id} perf stats: {self.get_stats()}")
                    print(f"This request took {processing_time:.4f}s")

                return APIResponse(
                    completion=completion,
                    request_id=request.request_id,
                    worker_id=self.worker_id,
                    processing_time=processing_time,
                    retry_count=retry_count,
                    final_attempt=True,
                )

            if int(os.getenv("DEBUG", "0")) >= 2:
                return await _call_api()

            try:
                return await _call_api()
            except Exception as e:
                processing_time = time.time() - start_time
                error_str = str(e)

                # Determine if this is a retryable error
                is_retryable = self._is_retryable_error(e)

                if is_retryable and retry_count < request.max_retries:
                    # Track retry attempt
                    self.retry_stats["total_retries"] += 1

                    # Calculate exponential backoff delay
                    delay = request.retry_delay * (2**retry_count) + random.uniform(
                        0, 1
                    )

                    if int(os.getenv("DEBUG", "0")) >= 2:
                        print(
                            f"Worker {self.worker_id}: Retrying request {request.request_id} "
                            f"(attempt {retry_count + 1}/{request.max_retries}) after {delay:.2f}s. "
                            f"ERROR: {error_str}"
                        )

                    # Wait before retry
                    await asyncio.sleep(delay)

                    # Recursive retry
                    result = await self._process_request_with_retry(
                        request, retry_count + 1
                    )

                    # Track successful retry if result is successful
                    if not result.error:
                        self.retry_stats["successful_retries"] += 1

                    return result
                else:
                    # Final failure or non-retryable error
                    if not is_retryable:
                        self.retry_stats["non_retryable_errors"] += 1
                    elif retry_count > 0:
                        self.retry_stats["failed_after_retries"] += 1
                        error_str = (
                            f"Failed after {retry_count + 1} attempts: {error_str}"
                        )

                    if int(os.getenv("DEBUG", "0")) >= 2:
                        print(
                            f"Worker {self.worker_id}: Failed request {request.request_id} with error: {error_str}"
                        )

                    return APIResponse(
                        completion="",
                        request_id=request.request_id,
                        worker_id=self.worker_id,
                        processing_time=processing_time,
                        error=error_str,
                        retry_count=retry_count,
                        final_attempt=True,
                    )

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        error_str = str(error).lower()

        # Retryable errors (transient issues)
        retryable_patterns = [
            "timeout",
            "timed out",
            "connection",
            "network",
            "503",  # Service Unavailable
            "502",  # Bad Gateway
            "504",  # Gateway Timeout
            "429",  # Rate Limited
            "500",  # Internal Server Error
            "rate limit",
            "quota exceeded",
            "overloaded",
            "unavailable",
            "temporary",
            "try again",
            "server error",
            "too many requests",
        ]

        # Non-retryable errors (permanent issues)
        non_retryable_patterns = [
            "400",  # Bad Request
            "401",  # Unauthorized
            "403",  # Forbidden
            "404",  # Not Found
            "invalid",
            "authentication",
            "permission",
            "malformed",
            "syntax error",
            "content policy",
            "safety",
        ]

        # Check for non-retryable patterns first
        for pattern in non_retryable_patterns:
            if pattern in error_str:
                return False

        # Check for retryable patterns
        for pattern in retryable_patterns:
            if pattern in error_str:
                return True

        # Default: retry unknown errors (conservative approach)
        return True

    async def process_logprob_request(self, request: LogprobRequest) -> LogprobResponse:
        """Process a logprob request with per-instance rate limiting and retry mechanism."""
        return await self._process_logprob_request_with_retry(request, 0)

    async def _process_logprob_request_with_retry(
        self, request: LogprobRequest, retry_count: int
    ) -> LogprobResponse:
        """Internal method to handle logprob request retries."""
        start_time = time.time()

        if int(os.getenv("DEBUG", "0")) >= 2:
            print(
                f"Processing logprob request {request.request_id} for {request.model_name}, retry {retry_count}"
            )

        if int(os.getenv("NO_RETRY", "0")):
            request.max_retries = 0
            request.retry_delay = 0
            retry_count = 0

        # Get or create semaphore for this specific instance's concurrency limit
        instance_key = f"{request.model_name}_{request.max_concurrent_per_worker}"
        if instance_key not in self.semaphores:
            self.semaphores[instance_key] = asyncio.Semaphore(
                request.max_concurrent_per_worker
            )

        instance_semaphore = self.semaphores[instance_key]

        async with instance_semaphore:
            try:
                # Only Together provider is supported for now
                if request.model_provider != "together":
                    raise NotImplementedError(
                        f"Logprobs not supported for {request.model_provider}"
                    )

                # Call the Together completion API directly
                result = await self._call_together_completion_directly(request)

                processing_time = time.time() - start_time
                self.requests_processed += 1
                self.total_processing_time += processing_time

                if int(os.getenv("DEBUG", "0")) >= 2:
                    print(
                        f"Completed logprob request {request.request_id} in {processing_time:.4f}s"
                    )

                return LogprobResponse(
                    logprobs=result["logprobs"],
                    request_id=request.request_id,
                    worker_id=self.worker_id,
                    processing_time=processing_time,
                    retry_count=retry_count,
                )

            except Exception as e:
                processing_time = time.time() - start_time
                error_str = str(e)

                # Determine if this is a retryable error
                is_retryable = self._is_retryable_error(e)

                if is_retryable and retry_count < request.max_retries:
                    # Track retry attempt
                    self.retry_stats["total_retries"] += 1

                    # Calculate exponential backoff delay
                    delay = request.retry_delay * (2**retry_count) + random.uniform(
                        0, 1
                    )

                    if int(os.getenv("DEBUG", "0")) >= 2:
                        print(
                            f"Worker {self.worker_id}: Retrying logprob request {request.request_id} "
                            f"(attempt {retry_count + 1}/{request.max_retries}) after {delay:.2f}s. "
                            f"ERROR: {error_str}"
                        )

                    # Wait before retry
                    await asyncio.sleep(delay)

                    # Recursive retry
                    result = await self._process_logprob_request_with_retry(
                        request, retry_count + 1
                    )

                    # Track successful retry if result is successful
                    if not result.error:
                        self.retry_stats["successful_retries"] += 1

                    return result
                else:
                    # Final failure or non-retryable error
                    if not is_retryable:
                        self.retry_stats["non_retryable_errors"] += 1
                    elif retry_count > 0:
                        self.retry_stats["failed_after_retries"] += 1
                        error_str = (
                            f"Failed after {retry_count + 1} attempts: {error_str}"
                        )

                    if int(os.getenv("DEBUG", "0")) >= 2:
                        print(
                            f"Worker {self.worker_id}: Failed logprob request {request.request_id} with error: {error_str}"
                        )

                    return LogprobResponse(
                        logprobs=0.0 if request.return_summed else [],
                        request_id=request.request_id,
                        worker_id=self.worker_id,
                        processing_time=processing_time,
                        error=error_str,
                        retry_count=retry_count,
                    )

    async def process_batch(self, requests: list[APIRequest]) -> list[APIResponse]:
        """Process a batch of requests concurrently."""
        return await asyncio.gather(
            *[self.process_request(request) for request in requests]
        )

    def get_stats(self) -> dict[str, Any]:
        """Get worker performance statistics."""
        avg_time = (
            self.total_processing_time / self.requests_processed
            if self.requests_processed > 0
            else 0.0
        )

        # Calculate retry success rate
        total_retries = self.retry_stats["total_retries"]
        retry_success_rate = (
            self.retry_stats["successful_retries"] / total_retries
            if total_retries > 0
            else 0.0
        )

        return {
            "worker_id": self.worker_id,
            "requests_processed": self.requests_processed,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_time,
            "requests_per_second": (
                self.requests_processed / self.total_processing_time
                if self.total_processing_time > 0
                else 0.0
            ),
            "retry_stats": {
                **self.retry_stats,
                "retry_success_rate": retry_success_rate,
            },
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.requests_processed = 0
        self.total_processing_time = 0.0
        self.retry_stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_after_retries": 0,
            "non_retryable_errors": 0,
        }


class RayAPIModel(APIModel):
    """
    Ray-based API model for maximum throughput on multicore servers.

    Key optimizations:
    - Shared worker pool across all instances
    - Connection pooling per worker
    - Intelligent load balancing
    - Batch processing with request pipelining
    - Minimized serialization overhead
    """

    # Shared worker pool across all instances
    _shared_workers = []
    _shared_workers_initialized = False
    _shared_worker_count = min(os.cpu_count(), int(os.getenv("MAX_WORKERS", "32")))
    _worker_lock = None  # Will be initialized when needed

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
        max_concurrent_per_worker: int = 100,
        batch_size: int = 50,
        enable_ray_logging: bool = False,
        default_max_retries: int = 3,
        default_retry_delay: float = 1.0,
        api_timeout: float = 600.0,  # Default 10 minute timeout
        few_shot_examples: list[dict[str, str]] = None,
        response_only: bool = True,  # Unused, kept for compatibility
    ):
        """
        Initialize Ray-based API model.

        :param model_name: The model name as it appears in the provider's API
        :param colloquial_name: Human-readable model name
        :param system_prompt: System prompt to prepend to conversations
        :param model_provider: API provider
        :param num_workers: Number of Ray workers (default: CPU count * 2)
        :param max_concurrent_per_worker: Max concurrent requests per worker
        :param batch_size: Batch size for request grouping
        :param enable_ray_logging: Enable Ray debug logging
        :param default_max_retries: Default number of retries for failed requests
        :param default_retry_delay: Default initial delay between retries in seconds
        :param api_timeout: Timeout in seconds for individual API calls
        """
        if int(os.getenv("USE_OPENROUTER", "0")) and model_provider == "auto":
            model_provider = "openrouter"

        super().__init__(
            model_name=model_name,
            colloquial_name=colloquial_name,
            system_prompt=system_prompt,
            model_provider=model_provider,
            few_shot_examples=few_shot_examples,
        )

        # Provider-specific optimization (only affects max_concurrent_per_worker now)
        if "openrouter" in model_provider:
            # OpenRouter models - use conservative limits since we're routing through their API
            if any(
                provider in model_name.lower()
                for provider in ["gpt", "openai", "gemini", "google"]
            ):
                max_concurrent_per_worker = (
                    100  # Conservative for high-volume models via OpenRouter
                )
            elif any(
                provider in model_name.lower() for provider in ["claude", "anthropic"]
            ):
                max_concurrent_per_worker = (
                    5  # Very conservative for Claude models via OpenRouter
                )
            else:
                max_concurrent_per_worker = (
                    20  # Moderate default for other OpenRouter models
                )
        elif "gpt" in model_name or "openai" in model_name or "gemini" in model_name:
            max_concurrent_per_worker = 300
        elif "claude" in model_name:
            max_concurrent_per_worker = 5
        else:
            max_concurrent_per_worker = int(
                os.getenv("MAX_CONCURRENT_PER_WORKER_DEFAULT", "5")
            )

        if int(os.getenv("DEBUG", "0")):
            enable_ray_logging = True

        # Instance configuration (workers are shared)
        self.max_concurrent_per_worker = max_concurrent_per_worker
        self.batch_size = batch_size
        self.enable_ray_logging = enable_ray_logging
        self.default_max_retries = default_max_retries
        self.default_retry_delay = default_retry_delay
        self.api_timeout = float(os.getenv("API_TIMEOUT", str(api_timeout)))

        if model_name.startswith("ft:"):
            # Is finetuned model, 10x timeout
            self.api_timeout = self.api_timeout * 10

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                num_cpus=os.cpu_count() // 4,
                num_gpus=0,
                log_to_driver=enable_ray_logging,
                ignore_reinit_error=True,
            )
            print(f"Ray initialized with {os.cpu_count()} CPUs and 0 GPUs")

        # Initialize shared worker pool if needed
        if not RayAPIModel._shared_workers_initialized:
            self._initialize_shared_workers()

        # Round-robin worker assignment
        self.current_worker_idx = 0

        # Performance tracking
        self.total_requests = 0
        self.total_batches = 0

    @classmethod
    def _initialize_shared_workers(cls):
        """Initialize the shared worker pool (called once)."""
        if cls._shared_workers_initialized:
            return

        print(
            f"Initializing shared Ray worker pool with {cls._shared_worker_count} workers..."
        )

        for i in range(cls._shared_worker_count):
            worker = APIWorker.remote(
                worker_id=f"shared_worker_{i}",
                model_name="shared",  # Will be overridden per request
                model_provider="auto",  # Will be overridden per request
                max_concurrent_requests=500,  # High default, controlled per instance
            )
            cls._shared_workers.append(worker)

        cls._shared_workers_initialized = True
        print(f"Shared worker pool initialized with {len(cls._shared_workers)} workers")

    def _get_next_worker(self):
        """Get next worker using round-robin load balancing from shared pool."""
        worker = RayAPIModel._shared_workers[self.current_worker_idx]
        self.current_worker_idx = (self.current_worker_idx + 1) % len(
            RayAPIModel._shared_workers
        )
        return worker

    def _create_request(
        self,
        history: list[dict[str, str]] | str,
        disable_system_prompt: bool = False,
        **kwargs,
    ) -> tuple[APIRequest, str]:
        """Create an API request from history."""
        if isinstance(history, str):
            history = [{"role": "user", "content": history}]
        else:
            history = deepcopy(history)

        if not disable_system_prompt and self.system_prompt:
            history.insert(0, {"role": "system", "content": self.system_prompt})

        prompt = Prompt(
            messages=[
                ChatMessage(
                    content=message["content"], role=MessageRole(message["role"])
                )
                for message in history
            ]
        )

        # Set default parameters
        if "temperature" not in kwargs:
            kwargs["temperature"] = self.temperature
        if "presence_penalty" not in kwargs:
            kwargs["presence_penalty"] = self.presence_penalty

        # Handle Claude-specific parameters
        if "claude" in self.model_name and "presence_penalty" in kwargs:
            if float(kwargs["presence_penalty"]) != 0:
                warnings.warn(
                    "Claude API doesn't support presence penalty. Ignoring the penalty."
                )
            del kwargs["presence_penalty"]

        # Validate dialogue structure
        for msg_idx in range(1, len(prompt.messages)):
            this_msg, last_msg = prompt.messages[msg_idx], prompt.messages[msg_idx - 1]
            is_valid = (
                this_msg.role.value in ("user", "assistant")
                and this_msg.role.value != last_msg.role.value
            )
            if not is_valid:
                raise ValueError(f"Malstructured LLM query: {prompt}")

        request_id = f"req_{self.total_requests}_{time.time()}"
        self.total_requests += 1

        # Extract retry parameters from kwargs
        max_retries = kwargs.pop("max_retries", self.default_max_retries)
        retry_delay = kwargs.pop("retry_delay", self.default_retry_delay)

        return (
            APIRequest(
                prompt=prompt,
                kwargs=kwargs,
                request_id=request_id,
                model_name=self.model_name,
                model_provider=self.model_provider,
                max_concurrent_per_worker=self.max_concurrent_per_worker,
                max_retries=max_retries,
                retry_delay=retry_delay,
                api_timeout=self.api_timeout,
            ),
            request_id,
        )

    async def infer_from_history_async(
        self,
        history: list[dict[str, str]] | str,
        disable_system_prompt: bool = False,
        disable_logging: bool = True,
        **kwargs,
    ) -> str:
        """Single inference using Ray workers."""
        history = self._prepend_few_shot_to_history(history)

        request, request_id = self._create_request(
            history, disable_system_prompt, **kwargs
        )

        # Log request if enabled (following APIModel pattern)
        if not disable_logging:
            dump_file(
                "data/tmp/___inference_record.jsonl",
                [
                    {"role": msg.role.value, "content": msg.content}
                    for msg in request.prompt.messages
                ],
                write_mode="a",
                indent=None,
            )

        # Send to next available worker
        worker = self._get_next_worker()
        response_future = worker.process_request.remote(request)
        response: APIResponse = await response_future

        if response.error:
            raise RuntimeError(f"API request failed: {response.error}")

        # Log response if enabled (following APIModel pattern)
        if not disable_logging:
            dump_file(
                "data/tmp/___inference_record.jsonl",
                response.completion,
                write_mode="a",
                indent=None,
            )

        return response.completion

    async def infer_from_histories_async(
        self,
        histories: list[list[dict[str, str]]] | list[str],
        disable_system_prompt: bool = False,
        **kwargs,
    ) -> list[str]:
        """Optimized batch inference using distributed Ray workers."""
        if not histories:
            return []

        histories = [
            self._prepend_few_shot_to_history(history) for history in histories
        ]

        # Create all requests
        requests = []
        request_ids = []
        for history in histories:
            request, request_id = self._create_request(
                history, disable_system_prompt, **kwargs
            )
            requests.append(request)
            request_ids.append(request_id)

        # Split requests into batches for each worker
        num_requests = len(requests)
        num_workers = len(RayAPIModel._shared_workers)
        requests_per_worker = num_requests // num_workers
        remainder = num_requests % num_workers

        worker_futures = []
        request_idx = 0

        for worker_idx, worker in enumerate(RayAPIModel._shared_workers):
            # Calculate batch size for this worker
            worker_batch_size = requests_per_worker
            if worker_idx < remainder:
                worker_batch_size += 1

            if worker_batch_size > 0:
                worker_requests = requests[
                    request_idx : request_idx + worker_batch_size
                ]
                request_idx += worker_batch_size

                # Send batch to worker
                future = worker.process_batch.remote(worker_requests)
                worker_futures.append(future)

        # Gather all responses using asyncio.gather with proper Ray futures
        worker_responses = await asyncio.gather(*worker_futures)

        # Flatten and sort responses by request_id to maintain order
        all_responses = []
        for worker_batch in worker_responses:
            all_responses.extend(worker_batch)

        # Sort by request_id to maintain original order
        response_map = {resp.request_id: resp for resp in all_responses}
        ordered_responses = [response_map[req_id] for req_id in request_ids]

        # Check for errors and extract completions
        completions = []
        for response in ordered_responses:
            if response.error:
                raise RuntimeError(f"Batch API request failed: {response.error}")
            completions.append(response.completion)

        self.total_batches += 1
        return completions

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.total_requests = 0
        self.total_batches = 0
        for worker in RayAPIModel._shared_workers:
            worker.reset_stats.remote()

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        # Gather stats from shared worker pool
        worker_futures = [
            worker.get_stats.remote() for worker in RayAPIModel._shared_workers
        ]
        worker_stats = ray.get(worker_futures)

        total_requests_processed = sum(
            stats["requests_processed"] for stats in worker_stats
        )
        total_processing_time = sum(
            stats["total_processing_time"] for stats in worker_stats
        )

        return {
            "model_name": self.model_name,
            "num_workers": len(RayAPIModel._shared_workers),
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "total_requests_processed": total_requests_processed,
            "total_processing_time": total_processing_time,
            "overall_throughput_rps": (
                total_requests_processed / total_processing_time
                if total_processing_time > 0
                else 0.0
            ),
            "worker_stats": worker_stats,
        }

    def shutdown(self):
        """Individual instances don't shutdown shared workers."""
        # Only clear individual instance state
        self.total_requests = 0
        self.total_batches = 0

    def supports_logprobs(self) -> bool:
        """Whether this policy supports logprobs."""
        return self.model_provider in ["together"]

    async def logprobs_single_async(
        self, dialogue: list[dict[str, str]], return_summed: bool = True, **kwargs
    ) -> float | list[float]:
        """Calculate log probabilities for a single dialogue using Ray workers.

        This method uses the APIWorker infrastructure to handle logprob requests,
        benefiting from concurrency limiting, retry logic, and load balancing.

        :param dialogue: OpenAI-format dialogue
        :param return_summed: If True, return sum of logprobs. If False, return list of per-token logprobs.
        :param kwargs: Additional keyword arguments
        :return: Sum of log probabilities or list of per-token logprobs (without token text, per schema)
        """
        if self.model_provider != "together":
            raise NotImplementedError(
                f"Logprobs are not supported for {self.model_provider} models in RayAPIModel."
            )

        dialogue = self._prepend_few_shot_to_history(dialogue)

        # Create a logprob request
        request_id = f"logprob_{self.total_requests}_{time.time()}"
        self.total_requests += 1

        # Extract retry parameters from kwargs
        max_retries = kwargs.pop("max_retries", self.default_max_retries)
        retry_delay = kwargs.pop("retry_delay", self.default_retry_delay)

        logprob_request = LogprobRequest(
            dialogue=dialogue,
            return_summed=return_summed,
            request_id=request_id,
            model_name=self.model_name,
            model_provider=self.model_provider,
            system_prompt=self.system_prompt,
            max_concurrent_per_worker=self.max_concurrent_per_worker,
            max_retries=max_retries,
            retry_delay=retry_delay,
            kwargs=kwargs,  # Pass additional kwargs like debug
            api_timeout=self.api_timeout,
        )

        # Send to next available worker
        worker = self._get_next_worker()
        response_future = worker.process_logprob_request.remote(logprob_request)
        response: LogprobResponse = await response_future

        if response.error:
            raise RuntimeError(f"Logprob calculation failed: {response.error}")

        return response.logprobs

    async def embed_async(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        Generate embeddings using Google Vertex AI or other providers.

        :param texts: List of text strings to embed.
        :type texts: list[str]
        :param kwargs: Additional keyword arguments (e.g., task_type for Vertex AI)
        :type kwargs: dict
        :return: List of embedding vectors, one per input text.
        :rtype: list[list[float]]
        """
        # Check if this is a Vertex AI embedding model
        if self.model_name == "gemini-embedding-001" or (
            self.model_provider == "google" and "embedding" in self.model_name.lower()
        ):
            return await self._embed_vertex_ai(texts, **kwargs)
        else:
            raise NotImplementedError(
                f"Embedding not supported for model {self.model_name} with provider {self.model_provider}"
            )

    async def _embed_vertex_ai(
        self, texts: list[str], task_type: str = "SEMANTIC_SIMILARITY", **kwargs
    ) -> list[list[float]]:  # noqa: ARG002
        """
        Generate embeddings using Google Vertex AI.

        :param texts: List of text strings to embed.
        :param task_type: Task type for the embedding (e.g., SEMANTIC_SIMILARITY, RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT)
        :return: List of embedding vectors.
        """
        try:
            # Load the embedding model
            model = TextEmbeddingModel.from_pretrained(self.model_name)

            # Generate embeddings with batching for efficiency
            batch_size = 250  # Vertex AI limit
            all_embeddings = []

            async def get_embeddings_for_batch(batch_inputs):
                for retry_idx in range(5):
                    retry_delay = 1.6**retry_idx
                    try:
                        return await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda batch_inputs=batch_inputs: model.get_embeddings(
                                batch_inputs
                            ),
                        )
                    except Exception as e:
                        if retry_idx < 4:
                            await asyncio.sleep(retry_delay)
                        else:
                            raise e from e

            embedding_futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Create embedding inputs with task type
                inputs = [
                    TextEmbeddingInput(text=text, task_type=task_type) for text in batch
                ]

                # Get embeddings
                embedding_futures.append(get_embeddings_for_batch(inputs))

            embedding_batches = await asyncio.gather(*embedding_futures)
            for embeddings in embedding_batches:
                # Extract embedding vectors
                for embedding in embeddings:
                    all_embeddings.append(embedding.values)

            return all_embeddings

        except ImportError as ie:
            raise ImportError(
                "Google Vertex AI SDK not installed. Install with: pip install google-cloud-aiplatform"
            ) from ie
        except Exception as e:
            raise RuntimeError(f"Vertex AI embedding failed: {e}") from e

    async def train_sft_async(
        self,
        samples: list,
        validation_samples: list = None,
        metadata: dict = {},  # noqa: B006
    ) -> "RayAPIModel":
        """
        Perform supervised fine-tuning using OpenAI or Together AI APIs.
        Inherits the implementation from APIModel.

        :param samples: List of training samples
        :param validation_samples: Optional list of validation samples
        :param metadata: Training metadata
        :return: New fine-tuned RayAPIModel instance
        """
        from utils.policy_utils import create_policy_from_string

        # Temporarily disable OpenRouter when finetuning
        new_model = create_policy_from_string(
            self.colloquial_name,
            use_openrouter=False,
            system_prompt=self.system_prompt,
            few_shot_examples=self.few_shot_examples,
        )

        # RayAPIModel extends APIModel, so we can use the parent implementation
        finetuned_model = await APIModel.train_sft_async(
            new_model, samples, validation_samples=validation_samples, metadata=metadata
        )

        # OpenRouter is not supported for finetuned models; use original model provider
        return RayAPIModel(
            model_name=finetuned_model.model_name,
            colloquial_name=finetuned_model.colloquial_name,
            system_prompt=self.system_prompt,
            model_provider=finetuned_model.model_provider,
            few_shot_examples=self.few_shot_examples,
        )

    async def train_rl_async(
        self,
        samples: list,
        grader,
        validation_samples: list = None,
        metadata: dict = {},  # noqa: B006
    ) -> "RayAPIModel":
        """
        Perform reinforcement learning using OpenAI's RL API.
        Inherits the implementation from APIModel.

        :param samples: List of EvaluatedSample training examples with rewards
        :param grader: Either a dict with full grader spec (for model or python grader) or a callable for python grader
        :param validation_samples: Optional list of validation samples
        :param metadata: Training metadata
        :return: New RL-trained RayAPIModel instance
        """
        from utils.policy_utils import create_policy_from_string

        # Temporarily disable OpenRouter when training
        new_model = create_policy_from_string(
            self.colloquial_name,
            use_openrouter=False,
            system_prompt=self.system_prompt,
            few_shot_examples=self.few_shot_examples,
        )

        # RayAPIModel extends APIModel, so we can use the parent implementation
        rl_model = await APIModel.train_rl_async(
            new_model,
            samples,
            grader=grader,
            validation_samples=validation_samples,
            metadata=metadata,
        )

        # OpenRouter is not supported for finetuned models; use original model provider
        return RayAPIModel(
            model_name=rl_model.model_name,
            colloquial_name=rl_model.colloquial_name,
            system_prompt=self.system_prompt,
            model_provider=rl_model.model_provider,
            few_shot_examples=self.few_shot_examples,
        )

    @classmethod
    def shutdown_shared_workers(cls):
        """Shutdown shared worker pool (call this when completely done)."""
        if cls._shared_workers_initialized:
            for worker in cls._shared_workers:
                ray.kill(worker)
            cls._shared_workers.clear()
            cls._shared_workers_initialized = False
            print("Shared Ray worker pool shutdown")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass
