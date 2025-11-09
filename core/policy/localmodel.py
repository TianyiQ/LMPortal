"""
Local model policy implementation using SGLang backend.

This module provides a LocalModel class that manages local LLM backends
using SGLang for robust inference and logprob calculation.
"""

import asyncio
import concurrent.futures
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Union

import requests
import torch

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    import wandb
except ImportError:
    wandb = None

from core.grader.schema import Grader
from core.policy.schema import Policy, SingleSample
from utils.io_utils import dump_file, logger

try:
    import sglang as sgl
    from nvitop import Device, GpuProcess, select_devices
except ImportError:
    warnings.warn("sglang and/or nvitop not available. LocalModel will not work.")
    sgl = None
    GpuProcess = None
    Device = None

try:
    import ray
except ImportError as e:
    raise ImportError("Ray is required for LocalModel. Install with: pip install ray") from e

try:
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
except ImportError as e:
    raise ImportError("SFT training requires trl. Install with: pip install trl") from e

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
except ImportError:
    warnings.warn("Accelerate not available. Multi-GPU training will not work optimally.")
    Accelerator = None
    set_seed = None

# Class-level semaphore for LocalModel concurrency (inference + training)
_LOCALMODEL_SEMAPHORE = None


def get_localmodel_semaphore():
    """Get or create the LocalModel semaphore for inference and training concurrency."""
    global _LOCALMODEL_SEMAPHORE
    if _LOCALMODEL_SEMAPHORE is None:
        max_concurrent = int(os.getenv("LOCALMODEL_MAX_CONCURRENT", "1"))
        _LOCALMODEL_SEMAPHORE = asyncio.Semaphore(max_concurrent)
    return _LOCALMODEL_SEMAPHORE


class SemaphoreStatus(int, Enum):
    """Status of the semaphore."""

    NOT_HELD = 0
    ATTEMPTING_HOLD = 1
    HELD = 2


@ray.remote
class LocalModelWorker:
    """Unified Ray worker for LocalModel inference and logprob computation."""

    def __init__(self):
        self.backend_port = None
        self.inference_fn = None
        self.logprob_fn = None
        self.use_api_mode = None
        self.session = None

    def initialize(self, backend_port: int, temperature: float, max_tokens: int, use_api_mode: bool = False):
        """Initialize connection to existing SGLang backend.

        Args:
            backend_port: Port where SGLang backend is running
            temperature: Default temperature for generation
            max_tokens: Default max tokens for generation
            use_api_mode: If True, use direct API calls instead of SGLang functions
        """
        import sglang as sgl

        self.backend_port = backend_port
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_api_mode = use_api_mode

        if use_api_mode:
            # Create a persistent requests session for this worker
            self.session = requests.Session()

            # Test connection to backend
            test_url = f"http://localhost:{backend_port}/health"
            response = self.session.get(test_url)
            if response.status_code != 200:
                raise RuntimeError(f"Backend health check failed: {response.status_code}")
            print(f"Worker connected to backend at port {backend_port}, health check status: {response.status_code}")
        else:
            # Original SGLang function mode
            # Connect to existing backend
            sgl.set_default_backend(sgl.RuntimeEndpoint(f"http://localhost:{backend_port}"))

            # Set up SGLang inference function
            @sgl.function
            def inference_function(s, conversation: list[dict[str, str]], temperature: float, max_tokens: int):
                for turn in conversation:
                    if turn["role"] == "system":
                        s += sgl.system(turn["content"])
                    elif turn["role"] == "user":
                        s += sgl.user(turn["content"])
                    elif turn["role"] == "assistant":
                        s += sgl.assistant(turn["content"])
                    else:
                        raise ValueError(f"Unknown role: {turn['role']}")

                s += sgl.assistant_begin()
                s += sgl.gen("response", max_tokens=max_tokens, temperature=temperature, return_logprob=False)

            # Set up SGLang logprob function
            @sgl.function
            def logprob_function(s, conversation: list[dict[str, str]]):
                # Handle None case during SGLang tracing
                if conversation is None:
                    conversation = [{"role": "user", "content": "dummy"}, {"role": "assistant", "content": "dummy"}]

                # Handle different conversation formats
                if isinstance(conversation, str):
                    conversation = [{"role": "assistant", "content": conversation}]
                elif not isinstance(conversation, list):
                    if hasattr(conversation, "__iter__") and not isinstance(conversation, (str, dict)):
                        conversation = list(conversation)
                    else:
                        raise ValueError(f"Unexpected conversation type: {type(conversation)}")

                for turn in conversation:
                    if isinstance(turn, str):
                        turn = {"role": "assistant", "content": turn}
                    elif not isinstance(turn, dict):
                        raise ValueError(f"Unexpected turn type: {type(turn)}")

                    if turn["role"] == "system":
                        s += sgl.system(turn["content"])
                    elif turn["role"] == "user":
                        s += sgl.user(turn["content"])
                    elif turn["role"] == "assistant":
                        s += sgl.assistant(turn["content"])
                    else:
                        raise ValueError(f"Unknown role: {turn['role']}")

                s += sgl.gen(
                    "logprobs", max_tokens=0, return_logprob=True, logprob_start_len=0, return_text_in_logprobs=True
                )

            self.inference_fn = inference_function
            self.logprob_fn = logprob_function

        return f"Worker initialized in {'API' if use_api_mode else 'SGLang'} mode"

    def process_inference_chunk(
        self, conversations: list[list[dict[str, str]]], temperature: float, max_tokens: int
    ) -> list[str]:
        """Process a chunk of conversations for inference."""
        if self.use_api_mode:
            if self.session is None:
                raise RuntimeError("Worker not properly initialized for API mode")
            return self._process_inference_chunk_api(conversations, temperature, max_tokens)
        else:
            return self._process_inference_chunk_sglang(conversations, temperature, max_tokens)

    def _process_inference_chunk_sglang(
        self, conversations: list[list[dict[str, str]]], temperature: float, max_tokens: int
    ) -> list[str]:
        """Process using SGLang functions (original implementation)."""
        if self.inference_fn is None:
            raise RuntimeError("Worker not initialized")

        batch_inputs = [
            {"conversation": conv, "temperature": temperature, "max_tokens": max_tokens} for conv in conversations
        ]

        # Use smaller progress bar threshold for chunks
        results = self.inference_fn.run_batch(batch_inputs, progress_bar=False)

        responses = []
        for result in results:
            if result and result.get_meta_info("response") is not None:
                responses.append(result["response"])
            else:
                responses.append("")  # Return empty string for failed cases

        return responses

    def _process_inference_chunk_api(
        self, conversations: list[list[dict[str, str]]], temperature: float, max_tokens: int
    ) -> list[str]:
        """Process using direct API calls to SGLang backend - plain sequential execution."""
        results = []

        for conv in conversations:
            url = f"http://localhost:{self.backend_port}/v1/chat/completions"

            # Prepare the request
            data = {
                "model": "default",  # SGLang uses "default" for the loaded model
                "messages": conv,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }

            try:
                response = self.session.post(url, json=data)
                if response.status_code == 200:
                    result = response.json()
                    results.append(result["choices"][0]["message"]["content"])
                else:
                    print(f"API error (status {response.status_code}): {response.text}")
                    results.append("")
            except Exception as e:
                print(f"Request failed: {e}")
                results.append("")

        return results

    def process_logprob_chunk(self, conversations: list[list[dict[str, str]]], return_summed: bool) -> list:
        """Process a chunk of conversations for logprob calculation."""
        if self.use_api_mode:
            if self.session is None:
                raise RuntimeError("Worker not properly initialized for API mode")
            return self._process_logprob_chunk_api(conversations, return_summed)
        else:
            return self._process_logprob_chunk_sglang(conversations, return_summed)

    def _process_logprob_chunk_sglang(self, conversations: list[list[dict[str, str]]], return_summed: bool) -> list:
        """Process logprobs using SGLang functions (original implementation)."""
        if self.logprob_fn is None:
            raise RuntimeError("Worker not initialized")

        batch_inputs = [{"conversation": conv} for conv in conversations]
        results = self.logprob_fn.run_batch(batch_inputs, progress_bar=False)

        all_results = []
        for result in results:
            if result is None:
                all_results.append(0.0 if return_summed else [])
                continue

            logprob_info = result.get_meta_info("logprobs")
            if logprob_info is not None and "input_token_logprobs" in logprob_info:
                token_logprobs = logprob_info["input_token_logprobs"]
                processed_logprobs = []

                for i, x in enumerate(token_logprobs):
                    logprob_value = None
                    token_text = None

                    if isinstance(x, (list, tuple)) and len(x) > 0:
                        if x[0] is not None:
                            logprob_value = float(x[0])
                        if len(x) > 2 and x[2] is not None:
                            token_text = str(x[2])
                        elif len(x) > 1 and x[1] is not None and isinstance(x[1], str):
                            token_text = str(x[1])
                    elif isinstance(x, dict) and "logprob" in x:
                        if x["logprob"] is not None:
                            logprob_value = float(x["logprob"])
                        if "text" in x and x["text"] is not None:
                            token_text = str(x["text"])
                        elif "token" in x and x["token"] is not None:
                            token_text = str(x["token"])
                    elif isinstance(x, (int, float)):
                        logprob_value = float(x)
                    elif isinstance(x, str):
                        logprob_value = float(x)

                    if logprob_value is not None:
                        if return_summed:
                            processed_logprobs.append(logprob_value)
                        else:
                            if token_text is not None:
                                processed_logprobs.append((logprob_value, token_text))
                            else:
                                processed_logprobs.append((logprob_value, f"<token_{i}>"))

                if return_summed:
                    all_results.append(float(sum(processed_logprobs)))
                else:
                    all_results.append(processed_logprobs)
            else:
                all_results.append(0.0 if return_summed else [])

        return all_results

    def _process_logprob_chunk_api(self, conversations: list[list[dict[str, str]]], return_summed: bool) -> list:
        """Process logprobs using direct API calls to SGLang backend - plain sequential execution."""
        results = []

        for conv in conversations:
            url = f"http://localhost:{self.backend_port}/generate"

            # Convert conversation to a single text prompt for the native API
            text_parts = []
            for msg in conv:
                if msg["role"] == "system":
                    text_parts.append(f"System: {msg['content']}\n")
                elif msg["role"] == "user":
                    text_parts.append(f"User: {msg['content']}\n")
                elif msg["role"] == "assistant":
                    text_parts.append(f"Assistant: {msg['content']}\n")

            text = "".join(text_parts)

            # Use native generate API with logprob parameters
            # Note: Parameters go at the top level, not in sampling_params
            data = {
                "text": text,
                "sampling_params": {
                    "max_new_tokens": 0,  # We don't want to generate new tokens
                    "temperature": 0.0,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": 0,  # Get logprobs for the entire input
                "return_text_in_logprobs": True,
            }

            try:
                response = self.session.post(url, json=data)
                if response.status_code == 200:
                    result = response.json()

                    # Extract logprobs from the native API response
                    if "meta_info" in result and "input_token_logprobs" in result["meta_info"]:
                        token_logprobs = result["meta_info"]["input_token_logprobs"]
                        token_texts = result["meta_info"].get("input_tokens", [])

                        processed_logprobs = []
                        for i, lp in enumerate(token_logprobs):
                            if lp is not None:
                                # Handle different formats of logprob data
                                logprob_value = None

                                if isinstance(lp, (int, float)):
                                    # Direct numeric value
                                    logprob_value = float(lp)
                                elif isinstance(lp, list) and len(lp) > 0:
                                    # List format - take first element if it's numeric
                                    if isinstance(lp[0], (int, float)):
                                        logprob_value = float(lp[0])
                                elif isinstance(lp, dict) and "logprob" in lp:
                                    # Dict format with logprob key
                                    logprob_value = float(lp["logprob"])
                                else:
                                    # Debug output for unexpected format
                                    if int(os.getenv("DEBUG", "0")) >= 2:
                                        print(f"Unexpected logprob format at index {i}: type={type(lp)}, value={lp}")
                                    continue

                                if logprob_value is not None:
                                    if return_summed:
                                        processed_logprobs.append(logprob_value)
                                    else:
                                        token_text = token_texts[i] if i < len(token_texts) else f"<token_{i}>"
                                        processed_logprobs.append((logprob_value, token_text))

                        if return_summed:
                            results.append(float(sum(processed_logprobs)) if processed_logprobs else 0.0)
                        else:
                            results.append(processed_logprobs)
                    else:
                        if int(os.getenv("DEBUG", "0")) >= 2:
                            print(f"No logprobs found in response: {result}")
                        results.append(0.0 if return_summed else [])
                else:
                    print(f"API error (status {response.status_code}): {response.text}")
                    if int(os.getenv("DEBUG", "0")) >= 2:
                        raise ValueError(f"API error (status {response.status_code}): {response.text}")
                    results.append(0.0 if return_summed else [])
            except Exception as e:
                print(f"Logprob request failed: {e}")

                if int(os.getenv("DEBUG", "0")) >= 2:
                    raise ValueError(f"Logprob request failed: {e}") from e

                results.append(0.0 if return_summed else [])

        return results

    def cleanup(self):
        """Clean up resources like the requests session."""
        if self.session:
            self.session.close()
            self.session = None


class LocalModel(Policy):
    """A local model policy using SGLang backend for inference."""

    # Class variable to track active instances (backend started but not stopped)
    _active_instances = 0
    _active_instances_lock = asyncio.Lock()

    def __init__(
        self,
        model_name: str,
        response_only: bool = False,
        colloquial_name: str | None = None,
        gpu_ids: list[int] | None = None,
        port: int | None = None,
        temperature: float = 0.25,
        max_tokens: int = 8192,
        username: str = "root",
        system_prompt: str | None = None,
        disable_reasoning: bool = False,
        few_shot_examples: list[dict[str, str]] = None,
    ):
        """
        Initialize a local model with SGLang backend.

        Args:
            model_name: Path or repo ID of the model
            response_only: Whether to only return the response, not the logprobs
            colloquial_name: Human-readable name for the model
            gpu_ids: List of GPU IDs to use for the model
            port: Port for the SGLang server (auto-assigned if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            username: Username for GPU process management
        """
        if sgl is None:
            raise ImportError("sglang is required for LocalModel. Install with: pip install sglang[all]")

        super().__init__(
            colloquial_name or LocalModel._extract_model_name(model_name),
            few_shot_examples=few_shot_examples,
        )

        self.model_name = model_name
        self.port = port or (13285 + random.randint(0, 2000))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.username = username
        self.response_only = response_only
        self.gpu_ids = gpu_ids or list(range(torch.cuda.device_count()))
        self.system_prompt = system_prompt
        self.disable_reasoning = int(os.getenv("DISABLE_REASONING", "0")) == 1 or disable_reasoning

        self.backend_process: subprocess.Popen | None = None
        self.backend_started = False
        self._gpu_count = len(self.gpu_ids)
        self._semaphore_held = SemaphoreStatus.NOT_HELD

        # SGLang functions will be defined after backend starts
        self._inference_fn = None
        self._logprob_fn = None

        # Ray worker pool
        self._ray_workers = None
        self._num_workers = min(os.cpu_count(), int(os.getenv("MAX_WORKERS", "256")))
        self._use_api_mode = int(os.getenv("LOCALMODEL_USE_API", "0")) == 1  # Use API mode if env var is set
        self._worker_last_used = {}  # Track when each worker was last used for LRU

    @staticmethod
    def _reasoning_model_type(model_name: str) -> str:
        """Determine the type of reasoning model."""
        if "Qwen3" in model_name and "instruct" not in model_name.lower():
            return "Qwen3"

        return None

    @staticmethod
    def _extract_model_name(model_name: str) -> str:
        """Extract a reasonable model name from the path."""
        name = model_name.split("/")[-1]
        if name.endswith("-Instruct"):
            name = name[:-9]
        return name

    @staticmethod
    def _get_per_gpu_memory() -> int:
        # Get all available GPUs
        devices = Device.cuda.all()
        memory_gbs = max(device.memory_total() // 1024**3 for device in devices)
        assert 5 < memory_gbs < 500, f"Weird Per-GPU Memory Size: {memory_gbs}GB"
        logger.minor("Total GPU per memory: {}GB", memory_gbs, dedup="message")
        return memory_gbs

    @staticmethod
    def _get_multi_gpu_config():
        """
        Detect multi-GPU setup and return appropriate configuration.

        Returns:
            dict: Configuration with keys:
                - num_gpus: Number of available GPUs
                - deepspeed_config: Path to DeepSpeed config or None
                - use_accelerate: Whether to use Accelerate
                - distributed_type: Type of distributed training
        """
        config = {"num_gpus": 1, "deepspeed_config": None, "use_accelerate": False, "distributed_type": "NO"}

        # Check for available GPUs
        if torch.cuda.is_available():
            config["num_gpus"] = torch.cuda.device_count()

        # Check for DeepSpeed configuration
        deepspeed_configs = ["data/config/deepspeed_zero2.json", "data/config/deepspeed_zero3.json"]

        for ds_config in deepspeed_configs:
            if Path(ds_config).exists():
                config["deepspeed_config"] = ds_config
                break

        # Determine if we should use Accelerate
        if config["num_gpus"] > 1:
            config["use_accelerate"] = True
            if config["deepspeed_config"]:
                config["distributed_type"] = "DEEPSPEED"
            else:
                config["distributed_type"] = "MULTI_GPU"
        # Don't use DeepSpeed for single GPU - it causes issues with device placement

        # Override with environment variables if set
        if os.getenv("FORCE_SINGLE_GPU", "0") == "1":
            config["num_gpus"] = 1
            config["use_accelerate"] = False
            config["distributed_type"] = "NO"

        if os.getenv("DISABLE_DEEPSPEED", "0") == "1":
            config["deepspeed_config"] = None
            if config["num_gpus"] > 1:
                config["distributed_type"] = "MULTI_GPU"
            else:
                config["distributed_type"] = "NO"

        return config

    @staticmethod
    def _get_training_batch_size_per_device(model_name: str) -> int:
        model_size = LocalModel._get_model_size(model_name)
        batch_size = LocalModel._get_per_gpu_memory() // (model_size * 2.5 + 7.5)
        return 2 ** math.floor(math.log2(batch_size))

    @staticmethod
    def _get_model_size(model_name: str) -> float:
        """Estimate model size in billions of parameters."""
        model_name_lower = model_name.lower()

        size_mapping = [
            ("17b-16e", 109),
            ("17b-128e", 400),
            ("v3", 685),
            ("405b", 405),
            ("235b", 235),
            ("70b", 70),
            ("30b", 30),
            ("27b", 27),
            ("24b", 24),
            ("13b", 13),
            ("12b", 12),
            ("1.7b", 1.7),
            ("1.5b", 1.5),
            ("0.6b", 0.6),
            ("0.5b", 0.5),
            ("9b", 9),
            ("8b", 8),
            ("7b", 7),
            ("4b", 4),
            ("3b", 3),
            ("2b", 2),
            ("1b", 1),
            ("360m", 0.360),
            ("135m", 0.135),
        ]

        for size_str, size_val in size_mapping:
            if size_str in model_name_lower:
                return size_val

        if int(os.getenv("DEBUG", "0")) >= 1:
            raise ValueError(f"Model {model_name} not found in size mapping")

        return 7  # Default assumption

    def _kill_gpu_processes(self, force: bool = True):
        """Kill existing GPU processes for this user."""
        if int(os.getenv("KILLALL", "1")) == 0:
            if int(os.getenv("DEBUG", "0")) >= 1:
                print("KILLALL is set to 0, so not killing GPU processes")
            return

        if Device is None:
            return

        try:
            devices = Device.cuda.all()
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)

            for device in devices:
                processes = device.processes()
                processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)

                for process in processes:
                    if process.username.lower() == self.username.lower():
                        try:
                            print(f"Killing GPU process {process.pid}: {process.cmdline}")
                            os.kill(process.pid, signal.SIGTERM)
                            time.sleep(0.5)
                            os.kill(process.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass  # Process already dead or permission denied
        except Exception as e:
            print(f"WARNING: Could not kill GPU processes: {e}")

        if force:
            subprocess.run(["utils/killall.sh"], check=True)
            time.sleep(15)

    def _start_backend(self, purpose: str = "responses") -> subprocess.Popen:
        """Start the SGLang backend server."""
        silent = os.environ.get("DEBUG", "0") == "0"
        if int(os.getenv("DEBUG", "0")) >= 2:
            print(f"Starting backend for {self.model_name} with purpose {purpose} in silent mode: {silent}")

        if os.environ.get("HALT_BEFORE_LOAD", "0") == "1":
            print("Halted before loading backend.")
            sys.exit(0)

        # Configuration based on purpose
        model_size = LocalModel._get_model_size(self.model_name)

        if purpose == "responses":
            frac_static = 0.8
            prefill_size = 8192
        else:
            frac_static = 0.6
            prefill_size = 1024

        # Build command arguments
        # Large model configuration (now unconditional)
        min_gpus_per_instance = (
            1
            if model_size <= 10
            else (
                2
                if model_size <= 30
                else 4 if model_size <= 80 else 8 if model_size <= 160 else 16 if model_size <= 320 else 32
            )
        )

        if (
            "fp8" in self.model_name.lower() or "deepseek" in self.model_name.lower()
        ):  # DeepSeek models are FP8 by default
            print(f"Model {self.model_name} is FP8, halving tensor parallelism")
            min_gpus_per_instance = (min_gpus_per_instance + 1) // 2

        if "select_devices" in globals():
            if select_devices(min_count=0, max_count=self._gpu_count, min_free_memory="130GiB"):
                print("GPU has 130GiB+ free memory per device, halving tensor parallelism")
                min_gpus_per_instance = (min_gpus_per_instance + 1) // 2

        while self._gpu_count % min_gpus_per_instance != 0:
            if self._gpu_count < min_gpus_per_instance and int(os.getenv("OVERRIDE_MIN_GPUS_PER_INSTANCE", "0")) == 0:
                raise ValueError(
                    f"Not enough GPUs ({self._gpu_count}) for model {self.model_name} with size {model_size}. Set OVERRIDE_MIN_GPUS_PER_INSTANCE=1 to override this check."
                )
            min_gpus_per_instance += 1

        assert self._gpu_count % min_gpus_per_instance == 0

        args = [
            "-m",
            "sglang.launch_server",
            "--port",
            str(self.port),
            "--tp",
            str(min_gpus_per_instance),
            "--dp",
            str(self._gpu_count // min_gpus_per_instance),
            "--model",
            self.model_name,
            "--mem-fraction-static",
            str(frac_static),
            "--chunked-prefill-size",
            str(prefill_size),
            "--trust-remote-code",
        ]

        # Add schedule conservativeness for small models
        if model_size <= 10:
            args.extend(["--schedule-conservativeness", "0.3"])

        if self.model_name == "Qwen/Qwen3-30B-A3B-Thinking-2507":
            args.extend(["--context-length", "262144"])

        if (
            "qwen3" in self.model_name.lower()
            and "instruct" not in self.model_name.lower()
            and "base" not in self.model_name.lower()
        ):
            args.extend(["--reasoning-parser", "qwen3"])

        if int(os.getenv("DEBUG", "0")) >= 2:
            print(
                f"Model size: {model_size}; GPU count: {self._gpu_count}; min_gpus_per_instance: {min_gpus_per_instance}"
            )

        # Model-specific configurations
        if "phi" in self.model_name.lower():
            args += ["--disable-flashinfer"]

        if "smol" in self.model_name.lower():
            args += ["--chat-template=chatml"]

        if "mislead" in self.model_name.lower():
            args += ["--chat-template=llama-2"]

        if "llama-4" in self.model_name.lower():
            args += ["--chat-template=llama-4"]

        print(f"Starting SGLang backend: {' '.join(args)} with silent={silent}")

        # Start the process
        stdout_redirect = subprocess.DEVNULL if silent else None
        stderr_redirect = subprocess.DEVNULL if silent else None
        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        executable = os.environ.get("PY_EXEC", "python")
        backend = subprocess.Popen(executable.split() + args, stdout=stdout_redirect, stderr=stderr_redirect, env=env)

        return backend

    def _wait_for_backend(self, max_retries: int = 200):
        """Wait for the backend to become available."""
        for attempt in range(max_retries):
            time.sleep(30)  # Wait 30 seconds between attempts

            try:
                print(f"Attempting to connect to backend (attempt {attempt + 1}/{max_retries})...")
                sgl.set_default_backend(sgl.RuntimeEndpoint(f"http://localhost:{self.port}"))

                # Test the connection with a simple request
                print("Successfully connected to backend!")
                return True

            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print("Retrying in 30 seconds...")
                continue

        raise RuntimeError(f"Failed to connect to backend after {max_retries} attempts")

    def _setup_sglang_functions(self):
        """Set up SGLang inference functions."""

        @sgl.function
        def inference_function(s, conversation: list[dict[str, str]], temperature: float, max_tokens: int):
            for turn in conversation:
                if turn["role"] == "system":
                    s += sgl.system(turn["content"])
                elif turn["role"] == "user":
                    s += sgl.user(turn["content"])
                elif turn["role"] == "assistant":
                    s += sgl.assistant(turn["content"])
                else:
                    raise ValueError(f"Unknown role: {turn['role']}")

            s += sgl.assistant_begin()
            s += sgl.gen("response", max_tokens=max_tokens, temperature=temperature, return_logprob=False)

        @sgl.function
        def logprob_function(s, conversation: list[dict[str, str]]):
            # Debug output to understand what SGLang is passing
            if int(os.getenv("DEBUG", "0")) >= 2:
                print("DEBUG: logprob_function called")
                print(f"DEBUG: s type: {type(s)}")
                print(f"DEBUG: conversation type: {type(conversation)}")
                print(f"DEBUG: conversation value: {conversation}")

            # Handle the case where conversation might be a string or other type
            if isinstance(conversation, str):
                # If conversation is a string, assume it's assistant content
                conversation = [{"role": "assistant", "content": conversation}]
            elif not isinstance(conversation, list):
                raise ValueError(f"Unexpected conversation type: {type(conversation)}")

            if int(os.getenv("DEBUG", "0")) >= 2:
                print(f"DEBUG: Final conversation type: {type(conversation)}")
                print(f"DEBUG: Final conversation length: {len(conversation)}")
                if conversation:
                    print(f"DEBUG: First turn: {conversation[0]}")

            for turn in conversation:
                # Additional safety check
                if isinstance(turn, str):
                    # If turn is a string, assume it's assistant content
                    turn = {"role": "assistant", "content": turn}
                elif not isinstance(turn, dict):
                    raise ValueError(f"Unexpected turn type: {type(turn)}")

                if turn["role"] == "system":
                    s += sgl.system(turn["content"])
                elif turn["role"] == "user":
                    s += sgl.user(turn["content"])
                elif turn["role"] == "assistant":
                    s += sgl.assistant(turn["content"])
                else:
                    raise ValueError(f"Unknown role: {turn['role']}")

            s += sgl.gen(
                "logprobs", max_tokens=0, return_logprob=True, logprob_start_len=0, return_text_in_logprobs=True
            )

        @sgl.function
        def test_connection_function(s):
            s += sgl.gen("test", max_tokens=1, temperature=0.1)

        self._inference_fn = inference_function
        self._logprob_fn = logprob_function
        self._test_connection_fn = test_connection_function

    async def _ensure_backend_ready(self):
        """Ensure the backend is started and ready."""
        if not self.backend_started:
            await self.start_backend("responses" if self.response_only else "logprobs")

    async def _ensure_ray_workers_ready(self):
        """Ensure Ray is initialized and workers are ready."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                num_cpus=os.cpu_count() // 4,
                num_gpus=0,
                log_to_driver=int(os.getenv("DEBUG", "0")) >= 2,
                ignore_reinit_error=True,
            )
            print(f"Ray initialized with {os.cpu_count()} CPUs and 0 GPUs")

        # Initialize worker pool if not already done
        if self._ray_workers is None:
            self._ray_workers = [LocalModelWorker.remote() for _ in range(self._num_workers)]

            # Initialize all workers to connect to the existing backend
            init_tasks = [
                worker.initialize.remote(self.port, self.temperature, self.max_tokens, self._use_api_mode)
                for worker in self._ray_workers
            ]
            init_results = await asyncio.gather(*[asyncio.to_thread(ray.get, task) for task in init_tasks])

            if int(os.getenv("DEBUG", "0")):
                print(
                    f"Initialized {self._num_workers} Ray workers in {'API' if self._use_api_mode else 'SGLang'} mode: {init_results}"
                )

    async def _parallel_process(self, items: list, process_method: str, *args, **kwargs) -> list:
        """Generic parallel processing using Ray workers.

        Args:
            items: List of items to process
            process_method: Name of the worker method to call
            *args: Additional arguments to pass to the worker method
            **kwargs: Additional keyword arguments

        Returns:
            List of results in the same order as input items
        """
        # Ensure workers are ready
        await self._ensure_ray_workers_ready()

        # In API mode, distribute individual items across workers for maximum parallelism
        if self._use_api_mode:
            # Each worker will process individual items sequentially
            # But we distribute items across all workers for parallelism
            futures = []
            for i, item in enumerate(items):
                worker = self._ray_workers[i % self._num_workers]
                method = getattr(worker, process_method)
                # Send single-item list to worker
                futures.append(method.remote([item], *args, **kwargs))

            # Gather results (each result is a single-item list)
            results = await asyncio.gather(*[asyncio.to_thread(ray.get, future) for future in futures])

            # Flatten single-item lists back to individual results
            return [result[0] for result in results]
        else:
            # For SGLang mode, assign entire batch to a single worker
            # SGLang handles parallelization within the batch efficiently

            # Find the least recently used worker
            current_time = time.time()

            # Initialize last used times if needed
            for i, worker in enumerate(self._ray_workers):
                if i not in self._worker_last_used:
                    self._worker_last_used[i] = 0

            # Select the LRU worker
            lru_worker_idx = min(self._worker_last_used.keys(), key=lambda k: self._worker_last_used[k])
            worker = self._ray_workers[lru_worker_idx]

            # Update last used time
            self._worker_last_used[lru_worker_idx] = current_time

            # Send the entire batch to the selected worker
            method = getattr(worker, process_method)
            future = method.remote(items, *args, **kwargs)

            # Get results from the single worker
            results = await asyncio.to_thread(ray.get, future)

            return results

    async def start_backend(self, purpose: str = "responses"):
        """Start the backend if not already started."""
        if self.backend_started or self._semaphore_held != SemaphoreStatus.NOT_HELD:
            return

        print(f"Starting backend for {self.model_name}...")

        # Get the semaphore for LocalModel concurrency
        semaphore = get_localmodel_semaphore()

        # Acquire semaphore and HOLD IT until stop_backend is called
        print(f"Waiting to acquire semaphore for {self.model_name}...")
        self._semaphore_held = SemaphoreStatus.ATTEMPTING_HOLD
        await semaphore.acquire()
        self._semaphore_held = SemaphoreStatus.HELD

        try:
            # Track active instances
            async with LocalModel._active_instances_lock:
                LocalModel._active_instances += 1
                max_concurrent = int(os.getenv("LOCALMODEL_MAX_CONCURRENT", "1"))
                print(
                    f"Acquired semaphore for backend {self.model_name} (active: {LocalModel._active_instances}/{max_concurrent}) - will hold until stopped"
                )

            # Kill any existing processes first (included after acquiring semaphore as requested)
            self._kill_gpu_processes()

            # Start the backend
            self.backend_process = self._start_backend(purpose)

            # Set up SGLang functions
            self._setup_sglang_functions()

            # Wait for it to be ready
            self._wait_for_backend()

            self.backend_started = True
            print(f"Backend ready for {self.model_name}, holding semaphore until stop_backend is called")

        except Exception as e:
            # If startup fails, release the semaphore
            if self._semaphore_held == SemaphoreStatus.HELD:
                semaphore.release()
                self._semaphore_held = SemaphoreStatus.NOT_HELD
                async with LocalModel._active_instances_lock:
                    LocalModel._active_instances = max(0, LocalModel._active_instances - 1)
            raise e

    async def stop_backend(self):
        """Stop the backend and clean up resources."""
        if not self.backend_started:
            return

        print(f"Stopping backend for {self.model_name}...")

        # Clean up Ray workers first
        if self._ray_workers:
            print(f"Stopping {len(self._ray_workers)} Ray workers...")
            try:
                for worker in self._ray_workers:
                    ray.kill(worker)
                self._ray_workers = None
            except Exception:
                pass

        # Stop backend process
        if self.backend_process:
            print(f"Stopping backend process for {self.model_name}")
            try:
                self.backend_process.terminate()
                await asyncio.sleep(2)
                self.backend_process.kill()
            except Exception:
                pass

            self.backend_process = None

        # Kill any remaining GPU processes (done before releasing semaphore as requested)
        self._kill_gpu_processes()
        self.backend_started = False

        # Now release the semaphore that we've been holding since start_backend
        if self._semaphore_held == SemaphoreStatus.HELD:
            semaphore = get_localmodel_semaphore()
            semaphore.release()
            self._semaphore_held = SemaphoreStatus.NOT_HELD

            # Track active instances
            async with LocalModel._active_instances_lock:
                LocalModel._active_instances = max(0, LocalModel._active_instances - 1)
                max_concurrent = int(os.getenv("LOCALMODEL_MAX_CONCURRENT", "1"))
                print(
                    f"Released semaphore for {self.model_name} (active: {LocalModel._active_instances}/{max_concurrent})"
                )

    def _preprocess_conversation(
        self, conversation: list[dict[str, str]], disable_reasoning: bool = False, apply_on_response: bool = False
    ) -> list[dict[str, str]]:
        """Preprocess the conversation to handle things like reasoning."""
        conversation = deepcopy(conversation)
        conversation = self._prepend_few_shot_to_history(conversation)
        reasoning_model_type = LocalModel._reasoning_model_type(self.model_name)
        if reasoning_model_type == "Qwen3" and disable_reasoning:
            for idx, turn in enumerate(conversation):
                if turn["role"] == "user" and (
                    idx == len(conversation) - 1
                    or (
                        conversation[idx + 1]["role"] == "assistant"
                        and (
                            "</think>" not in conversation[idx + 1]["content"]
                            or "<think>\n\n</think>\n\n" in conversation[idx + 1]["content"]
                        )
                    )
                ):
                    turn["content"] = "/no_think " + turn["content"].strip()

                elif turn["role"] == "assistant" and apply_on_response and "</think>" not in turn["content"]:
                    turn["content"] = "<think>\n\n</think>\n\n" + turn["content"].strip()

            if int(os.getenv("DEBUG", "0")) >= 1 and random.random() < 0.01:
                print("DEBUG: Preprocessed conversation. Saved to data/tmp/preprocessed_conversation.json")
                dump_file("data/tmp/preprocessed_conversation.json", conversation)

        return conversation

    def _postprocess_response(self, response: str, ignore_reasoning: bool = False) -> str:
        """Postprocess the response to handle things like reasoning."""
        if "Qwen3" in self.model_name and "instruct" not in self.model_name.lower() and ignore_reasoning:
            response = response.replace("<think>\n\n</think>\n\n", "")
            if "</think>" in response and ignore_reasoning and len(response.split("</think>")) == 2:
                response = response.split("</think>")[1].strip()

        return response.strip()

    async def infer_single_async(
        self, history: list[dict[str, str]], disable_system_prompt: bool = False, **kwargs
    ) -> str:
        """Generate a single response from the model.

        :param history: The dialogue history, in OpenAI format.
        :type history: list[dict[str, str]]
        :param disable_system_prompt: Whether to disable the system prompt.
        :type disable_system_prompt: bool
        :param kwargs: Additional keyword arguments, e.g. `disable_reasoning` for local models.
        :type kwargs: dict
        :return: The single response.
        :rtype: str
        """
        return (await self.infer_batch_async([history], disable_system_prompt, **kwargs))[0]

    async def infer_batch_async(
        self, histories: list[list[dict[str, str]]], disable_system_prompt: bool = False, **kwargs
    ) -> list[str]:
        """Generate responses for a batch of conversations using true parallelization with Ray.

        :param histories: The list of dialogue histories, in OpenAI format.
        :type histories: list[list[dict[str, str]]]
        :param disable_system_prompt: Whether to disable the system prompt.
        :type disable_system_prompt: bool
        :param kwargs: Additional keyword arguments, e.g. `disable_reasoning` for local models.
        :type kwargs: dict
        :return: The list of responses.
        :rtype: list[str]
        """
        await self._ensure_backend_ready()

        disable_reasoning = kwargs.get("disable_reasoning", self.disable_reasoning)

        # Preprocess all conversations
        conversations = []
        for history in histories:
            # Add few-shot examples if available
            conversation = deepcopy(history)
            if self.system_prompt:
                if conversation and conversation[0]["role"] == "system":
                    raise ValueError("System prompt already exists in conversation, cannot add another system prompt")
                conversation = [{"role": "system", "content": self.system_prompt}] + conversation
            if disable_system_prompt:
                conversation = [turn for turn in conversation if turn["role"] != "system"]
            conversation = self._preprocess_conversation(conversation, disable_reasoning, apply_on_response=False)
            conversations.append(conversation)

        # Extract parameters
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Use parallel processing with Ray workers
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Process in parallel across workers
                raw_responses = await self._parallel_process(
                    conversations, "process_inference_chunk", temperature, max_tokens
                )

                # Postprocess responses
                responses = []
                for raw_response in raw_responses:
                    if raw_response:
                        if (
                            disable_reasoning
                            and int(os.getenv("DEBUG", "0")) >= 1
                            and "</think>" in raw_response
                            and "<think>\n\n</think>\n\n" not in raw_response
                        ):
                            warnings.warn(
                                "Response contains reasoning. Saved to data/tmp/response_with_reasoning.txt",
                                stacklevel=2,
                            )
                            dump_file("data/tmp/response_with_reasoning.txt", raw_response)

                        responses.append(self._postprocess_response(raw_response, ignore_reasoning=disable_reasoning))
                    else:
                        responses.append("")

                return responses

            except Exception as e:
                print(f"Batch inference attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    raise RuntimeError(f"Batch inference failed after {max_retries} attempts: {e}") from e

    def supports_logprobs(self) -> bool:
        """
        Whether this policy supports logprobs.
        """
        return True

    async def logprobs_single_async(
        self, dialogue: list[dict[str, str]], return_summed: bool = True, **kwargs
    ) -> float | list[float] | list[tuple[float, str]]:
        """Calculate the log probabilities for the given dialogue.

        Args:
            dialogue: OpenAI-format dialogue, e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            return_summed: If True, return sum of logprobs. If False, return list of per-token logprobs with token text.
            kwargs: Additional keyword arguments, e.g. `disable_reasoning` for local models.

        Returns:
            Sum of log probabilities for the dialogue (if return_summed=True) or list of (logprob, token_text) tuples (if return_summed=False)
        """
        return (await self.logprobs_batch_async([dialogue], return_summed, **kwargs))[0]

    async def logprobs_batch_async(
        self, dialogues: list[list[dict[str, str]]], return_summed: bool = True, **kwargs
    ) -> list[float] | list[list[float]] | list[list[tuple[float, str]]]:
        """Calculate the log probabilities for a batch of dialogues using true parallelization with Ray.

        Args:
            dialogues: List of OpenAI-format dialogues
            return_summed: If True, return sums of logprobs. If False, return lists of per-token logprobs.
            kwargs: Additional keyword arguments, e.g. `disable_reasoning` for local models.

        Returns:
            List of sum of log probabilities for each dialogue (if return_summed=True) or list of per-token logprob lists (if return_summed=False)
        """
        disable_reasoning = kwargs.get("disable_reasoning", self.disable_reasoning)

        if self.response_only:
            raise ValueError("Logprob calculation is not supported for response-only models")

        # Ensure backend is started in main process first
        await self._ensure_backend_ready()

        # Preprocess all conversations
        conversations = []
        for dialogue in dialogues:
            conversation = deepcopy(dialogue)
            if self.system_prompt:
                if conversation and conversation[0]["role"] == "system":
                    raise ValueError("System prompt already exists in conversation, cannot add another system prompt")
                conversation = [{"role": "system", "content": self.system_prompt}] + conversation
            conversation = self._preprocess_conversation(conversation, disable_reasoning, apply_on_response=True)
            conversations.append(conversation)

        # Debug mode: return results immediately
        if int(os.getenv("DEBUG", "0")) >= 2:
            return await self._parallel_process(conversations, "process_logprob_chunk", return_summed)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Process in parallel across workers
                results = await self._parallel_process(conversations, "process_logprob_chunk", return_summed)
                return results

            except Exception as e:
                print(f"Batch logprob calculation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    raise RuntimeError(f"Batch logprob calculation failed after {max_retries} attempts: {e}") from e

    async def embed_async(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        Generate embeddings using SGlang's OpenAI-compatible embedding API.

        :param texts: List of text strings to embed.
        :type texts: list[str]
        :param kwargs: Additional keyword arguments
        :type kwargs: dict
        :return: List of embedding vectors, one per input text.
        :rtype: list[list[float]]
        """
        # Ensure backend is started
        await self._ensure_backend_ready()

        # Check if this is an embedding model
        if "embedding" not in self.model_name.lower():
            raise ValueError(
                f"Model {self.model_name} does not appear to be an embedding model. Use models like Qwen/Qwen3-Embedding-8B"
            )

        # Use SGlang's OpenAI-compatible embedding endpoint
        url = f"http://localhost:{self.port}/v1/embeddings"

        # Process in batches for efficiency
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Prepare request
            payload = {"model": self.model_name, "input": batch, "encoding_format": "float"}

            # Make request
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()

                result = response.json()

                # Extract embeddings
                for item in result["data"]:
                    all_embeddings.append(item["embedding"])

            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"SGlang embedding request failed: {e}") from e
            except (KeyError, ValueError) as e:
                raise RuntimeError(f"Failed to parse SGlang embedding response: {e}") from e

        return all_embeddings

    async def train_sft_async(
        self,
        samples: list[SingleSample],
        validation_samples: list[SingleSample] = None,
        metadata: dict = {},  # noqa: B006
    ) -> "LocalModel":
        """
        Train the model using supervised fine-tuning with trl and deepspeed.

        :param samples: List of training samples
        :param validation_samples: Optional list of validation samples
        :param metadata: Training metadata
        :return: A new LocalModel instance with the fine-tuned model
        """
        # Get the semaphore for LocalModel concurrency
        semaphore = get_localmodel_semaphore()

        # Use deep_copy first to create the new instance with proper naming
        trained_policy = self.deep_copy(
            suffix_type="sft",
            suffix_data=samples,
            metadata={
                "base_model": self.model_name,
                "training_samples": len(samples),
                "validation_samples": len(validation_samples) if validation_samples else 0,
                **metadata,
            },
        )

        # Get display name for logging
        model_display_name = (
            trained_policy.colloquial_name[:30] + "..."
            if len(trained_policy.colloquial_name) > 30
            else trained_policy.colloquial_name
        )

        # Get output directory from the new policy's name
        output_dir = Path("data/models") / trained_policy.colloquial_name

        # Prepare training data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(0, 1000000)}"
        train_file = output_dir / f"training_data_{timestamp}.jsonl"
        with open(train_file, "w", encoding="utf-8") as f:
            for sample in samples:
                # Convert SingleSample to chat format
                messages = sample.history + [{"role": "assistant", "content": sample.output}]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

        logger.major(f"[{model_display_name}] Prepared {len(samples)} training samples at {train_file}")

        # Prepare validation data if provided
        val_file = None
        if validation_samples:
            val_file = output_dir / f"validation_data_{timestamp}.jsonl"
            with open(val_file, "w", encoding="utf-8") as f:
                for sample in validation_samples:
                    messages = sample.history + [{"role": "assistant", "content": sample.output}]
                    f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

            logger.major(f"[{model_display_name}] Prepared {len(validation_samples)} validation samples at {val_file}")

        # Initialize WandB if API key is available and wandb is installed
        wandb_run = None
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key and wandb is not None:
            try:
                wandb_run = wandb.init(
                    project="truthseeking-local-sft",
                    name=f"{self.colloquial_name}-sft-{timestamp[:8]}",
                    config={
                        "model": self.model_name,
                        "num_train_samples": len(samples),
                        "num_val_samples": len(validation_samples) if validation_samples else 0,
                        "max_seq_length": self.max_tokens,
                        **metadata,
                    },
                    reinit=True,
                    dir="data/tmp/wandb",
                )
            except Exception as e:
                logger.major(f"Failed to initialize WandB: {e}")

        # Acquire semaphore to limit concurrent LocalModel operations (inference + training)
        print(f"[{model_display_name}] Waiting to acquire semaphore for SFT training...")
        await semaphore.acquire()

        try:
            # Track active instances
            async with LocalModel._active_instances_lock:
                LocalModel._active_instances += 1
                max_concurrent = int(os.getenv("LOCALMODEL_MAX_CONCURRENT", "1"))
                logger.major(
                    f"[{model_display_name}] Starting LocalModel SFT training (acquired semaphore, active: {LocalModel._active_instances}/{max_concurrent})"
                )

            # Run the actual training in a thread pool to avoid blocking
            loop = asyncio.get_running_loop()

            def _run_training(wandb_run=wandb_run, model_display_name=model_display_name):
                # Check if datasets library is available
                if load_dataset is None:
                    raise ImportError("datasets library is required for training. Install with: pip install datasets")

                # Load datasets
                train_dataset = load_dataset("json", data_files=str(train_file), split="train")
                eval_dataset = None
                if val_file:
                    eval_dataset = load_dataset("json", data_files=str(val_file), split="train")

                # Get multi-GPU configuration
                gpu_config = LocalModel._get_multi_gpu_config()
                logger.major(
                    f"[{model_display_name}] GPU Configuration: {gpu_config['num_gpus']} GPU(s), "
                    f"Distributed Type: {gpu_config['distributed_type']}"
                )

                # Adjust batch size based on GPU count and memory
                train_batch_size = LocalModel._get_training_batch_size_per_device(self.model_name)

                # With DeepSpeed or multi-GPU, we can use larger effective batch sizes
                if gpu_config["distributed_type"] == "DEEPSPEED":
                    # DeepSpeed handles gradient accumulation automatically
                    train_accumulation = max(1, 16 // (train_batch_size * gpu_config["num_gpus"]))
                elif gpu_config["num_gpus"] > 1:
                    # Multi-GPU: scale down accumulation since we have multiple GPUs
                    train_accumulation = max(1, 8 // (train_batch_size * gpu_config["num_gpus"]))
                else:
                    # Single GPU: original logic
                    train_accumulation = max(1, 8 // train_batch_size)

                effective_batch_size = train_batch_size * train_accumulation * gpu_config["num_gpus"]
                logger.major(
                    f"Running SFT with per-device batch size {train_batch_size}, "
                    f"accumulation steps {train_accumulation}, "
                    f"effective batch size {effective_batch_size}"
                )

                # Configure mixed precision based on GPU capabilities
                # Use bf16 if available (better for training stability), otherwise fp16
                use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                use_fp16 = not use_bf16 and torch.cuda.is_available()

                training_args = SFTConfig(
                    output_dir=str(output_dir),
                    num_train_epochs=2,
                    per_device_train_batch_size=train_batch_size,
                    per_device_eval_batch_size=train_batch_size,
                    gradient_accumulation_steps=train_accumulation,
                    eval_accumulation_steps=train_accumulation,
                    learning_rate=2e-5,
                    warmup_steps=100,
                    logging_steps=10,
                    max_length=2048,
                    save_strategy="steps" if eval_dataset else "epoch",
                    eval_strategy="steps" if eval_dataset else "no",
                    eval_steps=200 if eval_dataset else None,
                    save_steps=200 if eval_dataset else None,
                    bf16=use_bf16,
                    fp16=use_fp16,
                    deepspeed=gpu_config["deepspeed_config"] if gpu_config["num_gpus"] > 1 else None,
                    remove_unused_columns=False,
                    load_best_model_at_end=True if eval_dataset else False,
                    report_to=["wandb"] if wandb_run else [],
                    # Distributed training settings
                    ddp_find_unused_parameters=False if gpu_config["num_gpus"] > 1 else None,
                    dataloader_num_workers=4 if gpu_config["num_gpus"] > 1 else 2,
                    # SFT-specific parameters
                    dataset_text_field=None,  # Using messages format, not text field
                    packing=False,  # Can be enabled for efficiency
                    assistant_only_loss=False,  # Train on full sequence by default
                )

                if metadata.get("lora_rank"):
                    lora_config = LoraConfig(
                        r=metadata["lora_rank"],
                        lora_alpha=metadata["lora_rank"],
                        lora_dropout=0.1,
                        lora_train_bias=False,
                    )
                else:
                    lora_config = None

                # Custom callback to log comprehensive metrics
                class MetricsCallback:
                    def __init__(self, wandb_run):
                        self.wandb_run = wandb_run
                        self.train_losses = []
                        self.eval_losses = []

                    def on_log(self, args, state, control, logs=None, **kwargs):
                        if not self.wandb_run or not logs:
                            return

                        import numpy as np

                        metrics = {}

                        # Basic metrics
                        if "loss" in logs:
                            self.train_losses.append(logs["loss"])
                            metrics["train_loss"] = logs["loss"]
                            if len(self.train_losses) > 1:
                                metrics["train_loss_ci_lower"] = np.percentile(self.train_losses[-10:], 2.5)
                                metrics["train_loss_ci_upper"] = np.percentile(self.train_losses[-10:], 97.5)

                        if "eval_loss" in logs:
                            self.eval_losses.append(logs["eval_loss"])
                            metrics["val_loss"] = logs["eval_loss"]
                            if len(self.eval_losses) > 1:
                                metrics["val_loss_ci_lower"] = np.percentile(self.eval_losses[-5:], 2.5)
                                metrics["val_loss_ci_upper"] = np.percentile(self.eval_losses[-5:], 97.5)

                        # SFT-specific metrics from documentation
                        if "entropy" in logs:
                            metrics["entropy"] = logs["entropy"]

                        if "mean_token_accuracy" in logs:
                            metrics["mean_token_accuracy"] = logs["mean_token_accuracy"]

                        if "num_tokens" in logs:
                            metrics["num_tokens"] = logs["num_tokens"]

                        # Learning rate
                        if "learning_rate" in logs:
                            metrics["lr"] = logs["learning_rate"]

                        # Gradient norm
                        if "grad_norm" in logs:
                            metrics["grad_norm"] = logs["grad_norm"]

                        # Throughput metrics
                        if "train_samples_per_second" in logs:
                            metrics["throughput"] = logs["train_samples_per_second"]

                        # Epoch
                        if "epoch" in logs:
                            metrics["epoch"] = logs["epoch"]

                        # Step
                        metrics["step"] = state.global_step

                        # Log to WandB
                        self.wandb_run.log(metrics)

                        # Also log to console every 3rd logging step
                        if state.global_step % (args.logging_steps * 3) == 0:
                            status_msg = f"[{model_display_name}] Step {state.global_step}"
                            if "loss" in logs:
                                status_msg += f" | Train Loss: {logs['loss']:.4f}"
                            if "eval_loss" in logs:
                                status_msg += f" | Val Loss: {logs['eval_loss']:.4f}"
                            if "learning_rate" in logs:
                                status_msg += f" | LR: {logs['learning_rate']:.2e}"
                            logger.major(status_msg)

                # Create callback instance
                metrics_callback = MetricsCallback(wandb_run) if wandb_run else None

                # Create trainer with callbacks
                from transformers import TrainerCallback

                class CustomCallback(TrainerCallback):
                    def __init__(self, metrics_callback):
                        self.metrics_callback = metrics_callback

                    def on_log(self, args, state, control, logs=None, **kwargs):
                        if self.metrics_callback:
                            self.metrics_callback.on_log(args, state, control, logs, **kwargs)

                # Create trainer
                trainer = SFTTrainer(
                    model=self.model_name,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    args=training_args,
                    peft_config=lora_config,
                    callbacks=[CustomCallback(metrics_callback)] if metrics_callback else [],
                )

                # Train
                logger.major(f"[{model_display_name}] Training in progress...")
                trainer.train()

                # Log final metrics if WandB is active
                if wandb_run and metrics_callback:
                    try:
                        import numpy as np

                        final_metrics = {
                            "final_train_loss": (
                                np.mean(metrics_callback.train_losses[-10:]) if metrics_callback.train_losses else None
                            ),
                            "final_val_loss": (
                                np.mean(metrics_callback.eval_losses[-5:]) if metrics_callback.eval_losses else None
                            ),
                            "total_steps": trainer.state.global_step,
                            "total_epochs": trainer.state.epoch,
                        }
                        # Remove None values
                        final_metrics = {k: v for k, v in final_metrics.items() if v is not None}
                        if final_metrics:
                            wandb_run.log(final_metrics)
                    except Exception as e:
                        logger.major(f"Failed to log final metrics to WandB: {e}")

                # Save model
                trainer.save_model()

                logger.major(f"[{model_display_name}] Training completed, model saved to {output_dir}")

            # Run training in thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, _run_training)

        finally:
            # Always release the semaphore and decrement counter
            semaphore.release()
            async with LocalModel._active_instances_lock:
                LocalModel._active_instances = max(0, LocalModel._active_instances - 1)
                max_concurrent = int(os.getenv("LOCALMODEL_MAX_CONCURRENT", "1"))
                logger.major(
                    f"[{model_display_name}] Released SFT training semaphore (active: {LocalModel._active_instances}/{max_concurrent})"
                )

        # Update the trained_policy with the actual model path
        trained_policy.model_name = str(output_dir)

        # Save additional training info
        training_info = {
            "train_file": str(train_file),
            "val_file": str(val_file) if val_file else None,
            "training_completed": True,
        }
        dump_file(output_dir / f"training_info_{timestamp}.json", training_info, indent=2)

        # Finish WandB run if initialized
        if wandb_run and wandb is not None:
            try:
                wandb.finish()
            except Exception as e:
                logger.major(f"Failed to close WandB run: {e}")

        logger.major(f"[{model_display_name}] LocalModel training fully completed")
        return trained_policy

    async def train_rl_async(
        self,
        samples: list,  # List of Problem objects
        grader: Union[Grader, dict, Callable],
        validation_samples: list = None,
        metadata: dict = {},  # noqa: B006
    ) -> "LocalModel":
        """
        Train the model using reinforcement learning with GRPO.

        :param samples: List of Problem objects for training
        :param grader: Either a Grader instance, a dict with grader spec, or a callable for python grader
        :param validation_samples: Optional list of Problem objects for validation
        :param metadata: Training metadata
        :return: A new LocalModel instance with the RL-trained model
        """

        # Get the semaphore for LocalModel concurrency
        semaphore = get_localmodel_semaphore()

        # Use deep_copy first to create the new instance with proper naming
        trained_policy = self.deep_copy(
            suffix_type="rl",
            suffix_data=samples,
            metadata={
                "base_model": self.model_name,
                "training_method": "reinforcement_learning",
                "training_samples": len(samples),
                "validation_samples": len(validation_samples) if validation_samples else 0,
                **metadata,
            },
        )

        # Get display name for logging
        model_display_name = (
            trained_policy.colloquial_name[:30] + "..."
            if len(trained_policy.colloquial_name) > 30
            else trained_policy.colloquial_name
        )

        # Get output directory from the new policy's name
        output_dir = Path("data/models") / trained_policy.colloquial_name

        # Prepare training data for GRPO
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(0, 1000000)}"
        train_file = output_dir / f"rl_training_data_{timestamp}.jsonl"

        # Prepare grader using the new grader system
        from core.grader.schema import create_grader_from_spec

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
            raise ValueError(f"Invalid grader type: {type(grader)}. Must be a Grader instance, dict, or callable.")

        # Get the grader function for local execution
        # Check if this is a ModelGrader that might need async handling
        from core.grader.model_grader import ModelGrader

        is_model_grader = isinstance(grader_instance, ModelGrader)
        grader_type = grader_instance.to_openai_spec().get("type", "unknown")
        logger.major(f"[{model_display_name}] Using grader: {grader_type} type (is_model_grader: {is_model_grader})")

        # Prepare prompts dataset for GRPO (not pre-computed rewards!)
        # GRPO generates completions during training and uses the reward function
        import dataclasses

        with open(train_file, "w", encoding="utf-8") as f:
            for problem in samples:
                # Create user message from problem question
                prompt_messages = [{"role": "user", "content": problem.question}]

                # Store as a prompt for GRPO to complete
                data = {"prompt": prompt_messages}

                # Add all problem fields for the reward function to use
                if dataclasses.is_dataclass(problem):
                    problem_dict = dataclasses.asdict(problem)
                    # Add all fields except 'question' which is already in prompt
                    for key, value in problem_dict.items():
                        if key != "question" and key not in data:
                            data[key] = value
                else:
                    # If not a dataclass, add attributes directly
                    for attr in dir(problem):
                        if not attr.startswith("_") and attr != "question":
                            value = getattr(problem, attr, None)
                            if value is not None and not callable(value):
                                data[attr] = value

                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        logger.major(f"[{model_display_name}] Prepared {len(samples)} RL prompts at {train_file}")

        # Prepare validation data if provided
        val_file = None
        if validation_samples:
            val_file = output_dir / f"rl_validation_data_{timestamp}.jsonl"
            with open(val_file, "w", encoding="utf-8") as f:
                for problem in validation_samples:
                    prompt_messages = [{"role": "user", "content": problem.question}]
                    data = {"prompt": prompt_messages}

                    # Add all problem fields for validation
                    if dataclasses.is_dataclass(problem):
                        problem_dict = dataclasses.asdict(problem)
                        for key, value in problem_dict.items():
                            if key != "question" and key not in data:
                                data[key] = value
                    else:
                        for attr in dir(problem):
                            if not attr.startswith("_") and attr != "question":
                                value = getattr(problem, attr, None)
                                if value is not None and not callable(value):
                                    data[attr] = value

                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

            logger.major(
                f"[{model_display_name}] Prepared {len(validation_samples)} RL validation prompts at {val_file}"
            )

        # Initialize WandB if API key is available and wandb is installed
        wandb_run = None
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key and wandb is not None:
            try:
                wandb_run = wandb.init(
                    project="truthseeking-local-rl",
                    name=f"{self.colloquial_name}-rl-{timestamp[:8]}",
                    config={
                        "model": self.model_name,
                        "training_method": "grpo",
                        "num_train_samples": len(samples),
                        "num_val_samples": len(validation_samples) if validation_samples else 0,
                        "max_seq_length": self.max_tokens,
                        "grader_type": grader_type,
                        **metadata,
                    },
                    reinit=True,
                    dir="data/tmp/wandb",
                )
            except Exception as e:
                logger.major(f"Failed to initialize WandB: {e}")

        # Acquire semaphore to limit concurrent LocalModel operations (inference + training)
        print(f"[{model_display_name}] Waiting to acquire semaphore for RL training...")
        await semaphore.acquire()

        try:
            # Track active instances
            async with LocalModel._active_instances_lock:
                LocalModel._active_instances += 1
                max_concurrent = int(os.getenv("LOCALMODEL_MAX_CONCURRENT", "1"))
                logger.major(
                    f"[{model_display_name}] Starting LocalModel RL training with GRPO (acquired semaphore, active: {LocalModel._active_instances}/{max_concurrent})"
                )

            # Run the actual training in a thread pool to avoid blocking
            loop = asyncio.get_running_loop()

            def _run_rl_training(wandb_run=wandb_run, model_display_name=model_display_name):
                try:
                    if load_dataset is None:
                        raise ImportError(
                            "datasets library is required for training. Install with: pip install datasets"
                        )
                    from datasets import Dataset
                    from trl import GRPOConfig, GRPOTrainer

                    # Load training data
                    train_data = []
                    with open(train_file) as f:
                        for line in f:
                            train_data.append(json.loads(line))

                    # Create dataset with prompts for GRPO
                    train_dataset = Dataset.from_list(train_data)

                    # Load validation dataset if provided
                    eval_dataset = None
                    if val_file:
                        val_data = []
                        with open(val_file) as f:
                            for line in f:
                                val_data.append(json.loads(line))

                        eval_dataset = Dataset.from_list(val_data)

                    if int(os.getenv("DEBUG", "0")) >= 2:
                        import pdb
                        import sys
                        import traceback

                    # Create reward function that uses the grader
                    # This function will be called by GRPO during training to evaluate generated completions
                    def grpo_reward_function(prompts, completions, **kwargs):
                        """
                        Reward function for GRPO that uses our grader to score completions.

                        :param prompts: List of prompts (strings or message lists)
                        :param completions: List of generated completions (strings)
                        :param kwargs: Additional fields from the dataset (e.g., correct_option, options, aux_info, etc.)
                        :return: List of rewards (floats)
                        """
                        rewards = []

                        # Get optional fields from kwargs
                        correct_option = kwargs.get("correct_option", None)
                        options = kwargs.get("options", None)

                        # Validate that all lists have the same length
                        list_lengths = [len(prompts), len(completions)]
                        for key, value in kwargs.items():
                            if isinstance(value, list):
                                list_lengths.append(len(value))

                        if len(set(list_lengths)) > 1:
                            logger.major(
                                f"Warning: Inconsistent list lengths in grpo_reward_function: prompts={len(prompts)}, completions={len(completions)}, kwargs={{{', '.join(f'{k}={len(v) if isinstance(v, list) else type(v).__name__}' for k, v in kwargs.items())}}}"
                            )

                        for i, (prompt, completion) in enumerate(zip(prompts, completions, strict=False)):
                            # Apply the grader
                            try:
                                # Build aux_info from all available fields in kwargs
                                aux_info = {}

                                # Check if aux_info is directly provided in kwargs
                                aux_info_from_kwargs = kwargs.get("aux_info", None)
                                if (
                                    aux_info_from_kwargs is not None
                                    and isinstance(aux_info_from_kwargs, list)
                                    and i < len(aux_info_from_kwargs)
                                ):
                                    # If aux_info is provided as a list, use it as the base
                                    if isinstance(aux_info_from_kwargs[i], dict):
                                        aux_info = aux_info_from_kwargs[i].copy()
                                    else:
                                        aux_info = aux_info_from_kwargs[i]

                                # Add standard fields if they exist (these override aux_info if present)
                                if (
                                    correct_option is not None
                                    and isinstance(correct_option, list)
                                    and i < len(correct_option)
                                ):
                                    aux_info["correct_option"] = correct_option[i]
                                if options is not None and isinstance(options, list) and i < len(options):
                                    aux_info["options"] = options[i]

                                # Add any other list fields from kwargs (except aux_info which we already handled)
                                for k, v in kwargs.items():
                                    if (
                                        k not in ["correct_option", "options", "aux_info"]
                                        and isinstance(v, list)
                                        and i < len(v)
                                    ):
                                        aux_info[k] = v[i]

                                # Create the sample for grading
                                single_sample = SingleSample(
                                    history=(
                                        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
                                    ),
                                    output=(completion if isinstance(completion, str) else completion[0]["content"]),
                                    aux_info=aux_info,
                                )

                                # Handle async graders (like ModelGrader)
                                reward_future = asyncio.run_coroutine_threadsafe(
                                    grader_instance.grade_async(single_sample), loop
                                )
                                rewards.append(float(reward_future.result()))

                            except Exception as e:
                                logger.major(f"Grader failed for completion {i}: {e}")
                                if int(os.getenv("DEBUG", "0")) >= 2:
                                    extype, value, tb = sys.exc_info()
                                    traceback.print_exc()
                                    pdb.post_mortem(tb)

                                rewards.append(0.0)  # Default reward on failure

                        return rewards

                    # Get multi-GPU configuration
                    gpu_config = LocalModel._get_multi_gpu_config()
                    logger.major(
                        f"[{model_display_name}] RL GPU Configuration: {gpu_config['num_gpus']} GPU(s), "
                        f"Distributed Type: {gpu_config['distributed_type']}"
                    )

                    # GRPO training configuration with multi-GPU support
                    train_batch_size = LocalModel._get_training_batch_size_per_device(self.model_name)
                    train_accumulation = max(1, 8 // (train_batch_size * gpu_config["num_gpus"]))

                    # GRPO requires at least 2 generations per prompt for advantages calculation
                    num_generations = max(2, min(16, train_batch_size * gpu_config["num_gpus"]))

                    effective_batch_size = train_batch_size * train_accumulation * gpu_config["num_gpus"]

                    logger.major(
                        f"Running GRPO with per-device batch size {train_batch_size}, "
                        f"accumulation steps {train_accumulation}, num generations {num_generations}, "
                        f"effective batch size {effective_batch_size}, num_gpus {gpu_config['num_gpus']}"
                    )

                    # Configure mixed precision
                    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    use_fp16 = not use_bf16 and torch.cuda.is_available()

                    # Enable DeepSpeed when using accelerate launch with multi-GPU
                    use_deepspeed = (
                        gpu_config["distributed_type"] == "DEEPSPEED"
                        and gpu_config["num_gpus"] > 1
                        and gpu_config["deepspeed_config"] is not None
                    )

                    training_args = GRPOConfig(
                        output_dir=str(output_dir),
                        num_train_epochs=1,
                        per_device_train_batch_size=train_batch_size,
                        per_device_eval_batch_size=train_batch_size,
                        gradient_accumulation_steps=train_accumulation,
                        eval_accumulation_steps=train_accumulation,
                        learning_rate=1e-6,
                        warmup_steps=100,
                        logging_steps=10,
                        save_strategy="steps" if eval_dataset else "epoch",
                        eval_strategy="steps" if eval_dataset else "no",
                        eval_steps=200 if eval_dataset else None,
                        save_steps=200 if eval_dataset else None,
                        bf16=use_bf16,
                        fp16=use_fp16,
                        deepspeed=gpu_config["deepspeed_config"] if use_deepspeed else None,
                        remove_unused_columns=False,
                        load_best_model_at_end=True if eval_dataset else False,
                        report_to=["wandb"] if wandb_run else [],
                        # Distributed training settings
                        ddp_find_unused_parameters=False if gpu_config["num_gpus"] > 1 else None,
                        dataloader_num_workers=4 if gpu_config["num_gpus"] > 1 else 2,
                        # GRPO specific parameters
                        num_generations=num_generations,  # Number of generations per prompt
                        beta=0.0,  # Default to 0 as per documentation (KL term not essential)
                        num_iterations=1,  # Number of PPO iterations per batch
                        max_prompt_length=4096,
                        max_completion_length=4096,  # Max length of generated responses
                        scale_rewards="group",  # Scale rewards at group level
                        epsilon=0.2,  # PPO clipping range
                    )

                    if metadata.get("lora_rank"):
                        lora_config = LoraConfig(
                            r=metadata["lora_rank"],
                            lora_alpha=metadata["lora_rank"] * 2,
                            lora_dropout=0.1,
                        )
                    else:
                        lora_config = None

                    # Custom metrics callback for comprehensive logging
                    class RLMetricsCallback:
                        def __init__(self, wandb_run):
                            self.wandb_run = wandb_run
                            self.metrics_history = {"rewards": [], "entropy": [], "kl": [], "clip_ratio": []}

                        def on_log(self, args, state, control, logs=None, **kwargs):
                            if not self.wandb_run or not logs:
                                return

                            import numpy as np

                            metrics = {}

                            # GRPO-specific metrics
                            if "reward" in logs:
                                self.metrics_history["rewards"].append(logs["reward"])
                                metrics["mean_reward"] = logs["reward"]
                                if len(self.metrics_history["rewards"]) > 1:
                                    metrics["reward_ci_lower"] = np.percentile(
                                        self.metrics_history["rewards"][-10:], 2.5
                                    )
                                    metrics["reward_ci_upper"] = np.percentile(
                                        self.metrics_history["rewards"][-10:], 97.5
                                    )

                            if "reward_std" in logs:
                                metrics["reward_std"] = logs["reward_std"]

                            if "entropy" in logs:
                                self.metrics_history["entropy"].append(logs["entropy"])
                                metrics["entropy"] = logs["entropy"]

                            if "kl" in logs:
                                self.metrics_history["kl"].append(logs["kl"])
                                metrics["kl_divergence"] = logs["kl"]

                            if "clip_ratio/region_mean" in logs:
                                self.metrics_history["clip_ratio"].append(logs["clip_ratio/region_mean"])
                                metrics["clip_ratio"] = logs["clip_ratio/region_mean"]

                            if "completions/mean_length" in logs:
                                metrics["mean_completion_length"] = logs["completions/mean_length"]

                            if "learning_rate" in logs:
                                metrics["lr"] = logs["learning_rate"]

                            if "epoch" in logs:
                                metrics["epoch"] = logs["epoch"]

                            metrics["step"] = state.global_step
                            metrics["num_tokens"] = logs.get("num_tokens", 0)

                            # Log to WandB
                            self.wandb_run.log(metrics)

                            # Console logging every 3rd step
                            if state.global_step % (args.logging_steps * 3) == 0:
                                status_msg = f"[{model_display_name}] GRPO Step {state.global_step}"
                                if "reward" in logs:
                                    status_msg += f" | Reward: {logs['reward']:.4f}"
                                if "entropy" in logs:
                                    status_msg += f" | Entropy: {logs['entropy']:.4f}"
                                if "kl" in logs:
                                    status_msg += f" | KL: {logs['kl']:.4f}"
                                logger.major(status_msg)

                    # Create callback instance
                    metrics_callback = RLMetricsCallback(wandb_run) if wandb_run else None

                    # Create trainer callbacks
                    from transformers import TrainerCallback

                    class CustomRLCallback(TrainerCallback):
                        def __init__(self, metrics_callback):
                            self.metrics_callback = metrics_callback

                        def on_log(self, args, state, control, logs=None, **kwargs):
                            if self.metrics_callback:
                                self.metrics_callback.on_log(args, state, control, logs, **kwargs)

                    # Create GRPO trainer with reward function
                    # Let TRL handle model loading and distribution for both single and multi-GPU
                    trainer = GRPOTrainer(
                        model=self.model_name,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        args=training_args,
                        peft_config=lora_config,
                        reward_funcs=grpo_reward_function,  # Pass the reward function
                        callbacks=[CustomRLCallback(metrics_callback)] if metrics_callback else [],
                    )

                    # Train
                    logger.major(f"[{model_display_name}] RL training with GRPO in progress...")
                    trainer.train()

                    # Log final metrics if WandB is active
                    if wandb_run and metrics_callback:
                        try:
                            import numpy as np

                            final_metrics = {
                                "final_train_loss": (
                                    np.mean(metrics_callback.metrics_history["train_loss"][-10:])
                                    if metrics_callback.metrics_history["train_loss"]
                                    else None
                                ),
                                "final_val_loss": (
                                    np.mean(metrics_callback.metrics_history["eval_loss"][-5:])
                                    if metrics_callback.metrics_history["eval_loss"]
                                    else None
                                ),
                                "final_mean_reward": (
                                    np.mean(metrics_callback.metrics_history["rewards"][-10:])
                                    if metrics_callback.metrics_history["rewards"]
                                    else None
                                ),
                                "final_kl_div": (
                                    np.mean(metrics_callback.metrics_history["kl_div"][-10:])
                                    if metrics_callback.metrics_history["kl_div"]
                                    else None
                                ),
                                "total_steps": trainer.state.global_step,
                                "total_epochs": trainer.state.epoch,
                            }
                            # Remove None values
                            final_metrics = {k: v for k, v in final_metrics.items() if v is not None}
                            if final_metrics:
                                wandb_run.log(final_metrics)
                        except Exception as e:
                            logger.major(f"Failed to log final RL metrics to WandB: {e}")

                    # Save model
                    trainer.save_model()

                    logger.major(f"[{model_display_name}] RL training completed, model saved to {output_dir}")

                except ImportError as e:
                    logger.major(f"Failed to import required RL training libraries: {e}")
                    logger.major("Install with: pip install trl transformers datasets")
                    raise
                except Exception as e:
                    logger.major("RL training failed: {}", e)
                    raise

            # Run training in thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, _run_rl_training)

        finally:
            # Always release the semaphore and decrement counter
            semaphore.release()
            async with LocalModel._active_instances_lock:
                LocalModel._active_instances = max(0, LocalModel._active_instances - 1)
                max_concurrent = int(os.getenv("LOCALMODEL_MAX_CONCURRENT", "1"))
                logger.major(
                    f"[{model_display_name}] Released RL training semaphore (active: {LocalModel._active_instances}/{max_concurrent})"
                )

        # Update the trained_policy with the actual model path
        trained_policy.model_name = str(output_dir)

        # Save additional training info
        training_info = {
            "train_file": str(train_file),
            "val_file": str(val_file) if val_file else None,
            "training_method": "grpo",
            "grader_type": grader_type,
            "training_completed": True,
        }
        dump_file(output_dir / f"rl_training_info_{timestamp}.json", training_info, indent=2)

        # Finish WandB run if initialized
        if wandb_run and wandb is not None:
            try:
                wandb.finish()
            except Exception as e:
                logger.major(f"Failed to close WandB run: {e}")

        logger.major(f"[{model_display_name}] LocalModel RL training fully completed")
        return trained_policy

    def __del__(self):
        """Cleanup when the object is destroyed."""
        # Clean up Ray workers
        if hasattr(self, "_ray_workers") and self._ray_workers:
            try:
                for worker in self._ray_workers:
                    ray.kill(worker)
            except Exception:
                pass

        # Clean up backend
        if hasattr(self, "backend_started") and self.backend_started:
            # Note: We can't use async in __del__, so this is best effort
            try:
                if hasattr(self, "backend_process") and self.backend_process:
                    self.backend_process.terminate()
                    self.backend_process.kill()
                if hasattr(self, "_kill_gpu_processes"):
                    self._kill_gpu_processes()
            except Exception:
                pass

        # Release semaphore if we're still holding it
        if hasattr(self, "_semaphore_held") and self._semaphore_held == SemaphoreStatus.HELD:
            try:
                semaphore = get_localmodel_semaphore()
                semaphore.release()
                self._semaphore_held = SemaphoreStatus.NOT_HELD
            except Exception:
                pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_backend("responses" if self.response_only else "logprobs")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_backend()
