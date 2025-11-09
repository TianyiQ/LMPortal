import dataclasses
import datetime
import glob
import json
import os
import time
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional

from dacite import Config as DaciteConfig
from dacite import from_dict as dacite_from_dict


class ConsoleLogger:
    def __init__(self, save_to_file: bool = None):
        if save_to_file is None:
            save_to_file = bool(int(os.getenv("SAVE_TO_FILE", "0")))

        self.message_stem_window_size_secs = defaultdict(lambda: 600)
        self.message_stem_per_window_max_count = defaultdict(lambda: None)
        self.message_stem_timestamps = defaultdict(deque)

        self.message_window_size_secs = defaultdict(lambda: 600)
        self.message_per_window_max_count = defaultdict(lambda: None)
        self.message_timestamps = defaultdict(deque)

        self.debug_level = int(os.getenv("DEBUG", "0"))
        self.save_to_file = save_to_file
        if self.save_to_file:
            os.makedirs("data/tmp/logs", exist_ok=True)
            self.filename = f"data/tmp/logs/console-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{uuid.uuid4()}.log"
            self.file = open(self.filename, "w", encoding="utf-8")
        else:
            self.filename = None
            self.file = None

    def __del__(self):
        if self.save_to_file:
            self.file.close()

    @staticmethod
    def _truncate_str(s: str, len_trunc: int) -> str:
        if len(s) > len_trunc:
            return s[: len_trunc // 2] + "..." + s[-len_trunc // 2 :]
        return s

    def _log(
        self,
        level: int,
        message_stem: str,
        *args: Any,
        dedup: Literal["none", "message_stem", "message"] = "none",
        max_count: int = 1,
        len_trunc: int = 400,
        per_field_len_trunc: int = None,
        window_size_secs: int = 600,
        per_window_max_count: int = None,
        **kwargs: Any,
    ):
        if level > self.debug_level:
            return

        if self.save_to_file:
            # No truncation or deduplication when saving to file
            message_notrunc = message_stem.format(*args, **kwargs)
            self.file.write(
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message_notrunc}\n"
            )

        if per_field_len_trunc:
            args = list(args)
            for i, arg in enumerate(args):
                args[i] = self._truncate_str(str(arg), per_field_len_trunc)
            for key, value in kwargs.items():
                kwargs[key] = self._truncate_str(str(value), per_field_len_trunc)

        if args or kwargs:
            message = message_stem.format(*args, **kwargs)
        else:
            message = message_stem

        if window_size_secs is not None:
            self.message_stem_window_size_secs[message_stem] = window_size_secs
            self.message_window_size_secs[message] = window_size_secs
        if per_window_max_count is not None:
            self.message_stem_per_window_max_count[message_stem] = per_window_max_count
            self.message_per_window_max_count[message] = per_window_max_count

        current_time = time.time()

        while (
            self.message_timestamps[message]
            and self.message_timestamps[message][0]
            < current_time - self.message_window_size_secs[message]
        ):
            self.message_timestamps[message].popleft()
        while (
            self.message_stem_timestamps[message_stem]
            and self.message_stem_timestamps[message_stem][0]
            < current_time - self.message_stem_window_size_secs[message_stem]
        ):
            self.message_stem_timestamps[message_stem].popleft()

        if dedup == "message_stem":
            if len(self.message_stem_timestamps[message_stem]) > (
                self.message_stem_per_window_max_count.get(message_stem)
                or max_count
                or 1e10
            ):
                return
        elif dedup == "message":
            if len(self.message_timestamps[message]) > (
                self.message_per_window_max_count.get(message) or max_count or 1e10
            ):
                return
        elif dedup == "none":
            pass
        else:
            raise ValueError(f"Invalid dedup value: {dedup}")

        self.message_timestamps[message].append(current_time)
        self.message_stem_timestamps[message_stem].append(current_time)
        print(self._truncate_str(message, len_trunc))

    def minor(
        self,
        message_stem: str,
        *args: Any,
        dedup: Literal["none", "message_stem", "message"] = "none",
        max_count: int = 1,
        len_trunc: int = 400,
        per_field_len_trunc: int = None,
        window_size_secs: int = 600,
        per_window_max_count: int = None,
        **kwargs: Any,
    ):
        self._log(
            2,
            message_stem,
            *args,
            dedup=dedup,
            max_count=max_count,
            len_trunc=len_trunc,
            per_field_len_trunc=per_field_len_trunc,
            window_size_secs=window_size_secs,
            per_window_max_count=per_window_max_count,
            **kwargs,
        )

    def major(
        self,
        message_stem: str,
        *args: Any,
        dedup: Literal["none", "message_stem", "message"] = "none",
        max_count: int = 1,
        len_trunc: int = 400,
        per_field_len_trunc: int = None,
        window_size_secs: int = 600,
        per_window_max_count: int = None,
        **kwargs: Any,
    ):
        self._log(
            1,
            message_stem,
            *args,
            dedup=dedup,
            max_count=max_count,
            len_trunc=len_trunc,
            per_field_len_trunc=per_field_len_trunc,
            window_size_secs=window_size_secs,
            per_window_max_count=per_window_max_count,
            **kwargs,
        )

    def urgent(
        self,
        message_stem: str,
        *args: Any,
        dedup: Literal["none", "message_stem", "message"] = "none",
        max_count: int = 1,
        len_trunc: int = 400,
        per_field_len_trunc: int = None,
        window_size_secs: int = 600,
        per_window_max_count: int = None,
        **kwargs: Any,
    ):
        self._log(
            0,
            message_stem,
            *args,
            dedup=dedup,
            max_count=max_count,
            len_trunc=len_trunc,
            per_field_len_trunc=per_field_len_trunc,
            window_size_secs=window_size_secs,
            per_window_max_count=per_window_max_count,
            **kwargs,
        )


logger = ConsoleLogger()


def get_common_attr(objs: Any, attr: str, force_existence: bool = False) -> Any:
    """Get the common attribute value from a list of objects or a single object."""
    if not isinstance(objs, list):
        return objs.__getattribute__(attr)
    if not force_existence and not all(hasattr(obj, attr) for obj in objs):
        return None
    if any(
        objs[i].__getattribute__(attr) != objs[0].__getattribute__(attr)
        for i in range(len(objs))
    ):
        return None
    return objs[0].__getattribute__(attr)


def normalize_key(k: str) -> str:
    """Normalize keys to Title Case properly."""
    if "_" in k:
        # Convert snake_case to Title Case
        return " ".join(word.capitalize() for word in k.split("_"))
    elif " " in k:
        # Ensure each word is capitalized
        return " ".join(word.capitalize() for word in k.split(" "))
    else:
        # Single word - capitalize first letter
        return k.capitalize()


def _shared_prefix_len(sa: str, sb: str, *args: Optional[list[str]]) -> int:
    if args:
        ans = _shared_prefix_len(sa, sb)
        for s in args:
            ans = _shared_prefix_len(sa[:ans], s)
        return ans

    for i in range(min(len(sa), len(sb))):
        if sa[i] != sb[i]:
            return i
    return min(len(sa), len(sb))


def _shared_suffix_len(*args: list[str]) -> int:
    return _shared_prefix_len(*[s[::-1] for s in args])


def _shared_substr_len(sa: str, sb: str) -> int:
    if not sa or not sb:
        return 0
    return max(
        _shared_prefix_len(sa[i:], sb[j:])
        for i in range(len(sa))
        for j in range(len(sb))
    )


def concate_first_letters(s: str) -> str:
    return "".join(c[0] for c in s.split("_"))


def should_skip_file(filepath: str | Path) -> bool:
    if isinstance(filepath, Path):
        filepath = filepath.as_posix()

    blacklist = [".ipynb_checkpoints", ".DS_Store", "__pycache__"]
    return any(pattern in filepath for pattern in blacklist)


def get_documented_env_vars() -> dict:
    """Get a dictionary of documented environment variables from README."""
    documented_vars = [
        # API and Model Configuration
        "TEMPERATURE",
        "PRESENCE_PENALTY",
        "NO_RETRY",
        "USE_OPENROUTER",
        "USE_RAY",
        "USE_BATCH",
        "SYSTEM_PROMPT",
        # OpenRouter Configuration
        "OPENROUTER_API_KEY",
        # Local Model Configuration
        "DEBUG",
        "KILLALL",
        "HALT_BEFORE_LOAD",
        "OVERRIDE_MIN_GPUS_PER_INSTANCE",
        "PY_EXEC",
        # Ray and Distributed Computing
        "MAX_WORKERS",
        "MAX_CONCURRENT_PER_WORKER_DEFAULT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        # Data Processing and Output Control
        "RUNS_SUBDIR",
        "DIR_NAME",
        "NO_FILE_LOGGING",
        "SHOW_PROGRESS",
        "PARALLEL_BATCH",
        "TOTAL_ITERS",
        # Algorithm Configuration
        "DISABLE_SYSTEM_PROMPT_IN_BELIEF_MEASUREMENT",
        "USE_FIXED_JUDGE",
        "OBJECTIVE_BELIEF",
        "USE_PER_TRAJ_BELIEF_MEASURE",
        "QUALITATIVE_JUDGE_USE_FEW_SHOT",
        "USE_SINGLE_INFERENCE",
        "RUN_ID_SUFFIX",
        # Experimental Configuration
        "ALGO_NAME",
        "ALGO_NAMES",
        "DOMAIN_NAMES",
        "REASONING_MODE_NAMES",
        "LEGACY_POLICY_LIST",
        "FORBIDDEN_MODELS",
        "JUDGE_POLICY_NAMES",
        "NUM_TRAJECTORIES",
        "RERUN_INCOMPLETE",
        "RECOMPUTE_RESULTS",
        "WITL_RECOMPUTE_POLICY",
        # Trajectory-Belief Decoupling System
        "DECOUPLE_TRAJECTORY_BELIEFS",
        "RECOMPUTE_TRAJECTORIES",
        "RECOMPUTE_BELIEFS",
        # World-in-the-Loop Forecaster Templates
        "FORECASTER_TEMPLATE",
        # Hardware and Performance
        "CUDA_VISIBLE_DEVICES",
        "TOKENIZERS_PARALLELISM",
        "TORCH_NCCL_AVOID_RECORD_STREAMS",
        # Caching and Storage
        "NO_CACHE",
        "CACHE_DIR",
        "PROMPT_HISTORY_DIR",
        # Debug and Logging
        "VERL_LOGGING_LEVEL",
        "SAFETYTOOLING_PRINT_PROMPTS",
        "RAY_DEDUP_LOGS",
    ]

    result = {}
    for var in documented_vars:
        value = os.environ.get(var)
        if value is not None:
            # Skip API keys and tokens completely
            if "API" in var or "GOOGLE" in var or "TOKEN" in var:
                continue  # Don't include at all
            else:
                result[var] = value

    return result


def create_file_metadata(additional_metadata: dict = None) -> dict:
    """Create metadata to be included in JSON files."""
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "environment_variables": get_documented_env_vars(),
    }

    if additional_metadata:
        metadata.update(additional_metadata)

    return metadata


def strip_metadata_from_data(data: Any) -> Any:
    """Remove metadata fields from loaded data for backward compatibility."""
    if isinstance(data, dict):
        # Handle new two-field structure
        if "metadata" in data and "saved_content" in data and len(data) == 2:
            return data["saved_content"]
        # Handle legacy _metadata structure
        elif "_metadata" in data:
            data = {k: v for k, v in data.items() if k != "_metadata"}
    elif isinstance(data, list):
        # For lists, check if any items have metadata and strip it
        data = [strip_metadata_from_data(item) for item in data]

    return data


def extract_algorithm_name_from_run_id(run_id: str) -> str:
    """
    DEPRECATED: Extract algorithm name from run_id format.
    This function is kept for backward compatibility with old run IDs that include algorithm names.
    For new run IDs (without algorithm names), this will raise an error.
    """
    parts = run_id.split("-")
    if len(parts) >= 5:
        return parts[-1]  # Last part is the algorithm name (old format only)
    else:
        if os.environ.get("ALGO_NAME"):
            return os.environ["ALGO_NAME"]
        raise ValueError(
            f"Invalid run_id format or new format without algorithm name: {run_id}"
        )


def get_list_of_runs(subdir: str = "", skip_incomplete: bool = True) -> list[str]:
    """Return list of run IDs present in the runs folder

    :param subdir: _description_, defaults to ""
    :type subdir: str, optional
    :param skip_incomplete: Whether to skip runs that don't have all the files, defaults to True
    :type skip_incomplete: bool, optional
    :return: _description_
    :rtype: list[str]
    """

    # Path from which to start looking for runs
    # Check for subdirectory from environment variables (DIR_NAME takes precedence)
    runs_subdir = os.environ.get("RUNS_SUBDIR", os.environ.get("DIR_NAME", ""))
    if runs_subdir and not subdir:
        dir = os.path.join("data/runs/", runs_subdir)
    else:
        dir = os.path.join("data/runs/", subdir)

    # Get list of run IDs
    run_ids = [
        f.name.split("run-")[1]
        for f in os.scandir(dir)
        if f.is_dir() and "run-" in f.name
    ]

    # Skip incomplete runs if requested
    if skip_incomplete:
        complete_run_ids = []
        for run_id in run_ids:
            run_dir = os.path.join(dir, f"run-{run_id}")

            # Look for any bias-eval-results-*.json files
            bias_files = glob.glob(os.path.join(run_dir, "bias-eval-results-*.json"))

            # Also check for old format for backward compatibility
            old_format_file = os.path.join(run_dir, "bias-eval-results.json")

            if bias_files or os.path.exists(old_format_file):
                complete_run_ids.append(run_id)

        run_ids = complete_run_ids

    return run_ids


def complete_path(filepath: str, run_id: str, example_path: str = None) -> str:
    """
    Complete the path to the data directory and the run folder.
    Supports RUNS_SUBDIR and DIR_NAME environment variables for specifying subdirectories.
    DIR_NAME takes precedence over RUNS_SUBDIR for compatibility with batch processing scripts.
    """
    if filepath is None:
        return None

    if os.path.exists(filepath):
        return filepath

    if run_id not in filepath:
        filepath = os.path.join(f"run-{run_id}", filepath)

    if "runs/" not in filepath:
        # Check for subdirectory from environment variables (DIR_NAME takes precedence)
        if example_path:
            path_parts = example_path.split("run-")
            assert (
                len(path_parts) == 2
            ), f"Expected exactly one run directory in {example_path}, got {path_parts}"
            runs_subdir = path_parts[0]
            assert os.path.exists(
                runs_subdir
            ), f"Run directory {runs_subdir} does not exist"
        else:
            runs_subdir = os.environ.get("RUNS_SUBDIR", os.environ.get("DIR_NAME", ""))

        if runs_subdir.startswith("data/"):
            runs_subdir = runs_subdir[len("data/") :]

        if runs_subdir.startswith("runs/"):
            runs_subdir = runs_subdir[len("runs/") :]

        if runs_subdir:
            filepath = os.path.join("runs", runs_subdir, filepath)
        else:
            filepath = os.path.join("runs", filepath)

    if "data/" not in filepath:
        filepath = os.path.join("data", filepath)

    return filepath


def dump_file(
    filepath: str | Path,
    data: Any,
    write_mode: Literal["w", "a"] = "w",
    indent: Optional[int] = 2,
    **kwargs,
) -> None:
    """
    Dump a JSON object or dataclass instance to the data directory.
    This is the original version for backward compatibility with non-run-specific files.
    """
    if isinstance(filepath, Path):
        filepath = filepath.as_posix()

    data = deepcopy(data)
    if os.path.dirname(filepath) == "":
        filepath = os.path.join("data", filepath)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if dataclasses.is_dataclass(data):
        data = dataclasses.asdict(data)

    if isinstance(data, list):
        for i in range(len(data)):
            if dataclasses.is_dataclass(data[i]):
                data[i] = dataclasses.asdict(data[i])

    if write_mode == "a" and not os.path.exists(filepath):
        write_mode = "w"

    with open(filepath, write_mode, encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, **kwargs)
        if write_mode == "a":
            f.write("\n")


def load_file(
    filepath: str, schema: type | None = None, none_on_error: bool = False, **kwargs
) -> Any:
    """
    Load a JSON file or dataclass instance from the data directory.
    This is the original version for backward compatibility with non-run-specific files.
    """
    try:
        if not os.path.exists(filepath):
            if os.path.exists(os.path.join("data", filepath)):
                filepath = os.path.join("data", filepath)
            else:
                raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, encoding="utf-8") as file:
            data = json.load(file, **kwargs)

        # Handle legacy metadata format first (before stripping metadata)
        if isinstance(data, dict) and "data" in data and "_metadata" in data:
            data = data["data"]

        # Strip metadata for backward compatibility
        data = strip_metadata_from_data(data)

        if schema is None:
            return data
        elif dataclasses.is_dataclass(schema):
            config = DaciteConfig(check_types=False)

            if isinstance(data, list):
                data = [
                    dacite_from_dict(data_class=schema, data=item, config=config)
                    for item in data
                ]
            else:
                data = dacite_from_dict(data_class=schema, data=data, config=config)
        else:
            raise ValueError(f"Invalid schema: {schema}")
    except Exception as e:
        if none_on_error:
            return None
        else:
            raise e from e

    return data


def load_file_for_run(
    filepath: str, run_id: str, schema: type | None = None, example_path: str = None
) -> Any:
    """
    Load a JSON file or dataclass instance from the data directory for a specific run.
    """
    filepath = complete_path(filepath, run_id=run_id, example_path=example_path)
    return load_file(filepath, schema=schema)


def dump_file_for_run(
    filepath: str,
    data: Any,
    run_id: str,
    write_mode: Literal["w", "a"] = "w",
    indent: Optional[int] = 2,
    include_metadata: bool = True,
) -> None:
    """
    Dump a JSON object or dataclass instance to the data directory for a specific run.
    Automatically includes metadata (timestamp and environment variables) for tracking.
    """
    filepath = complete_path(filepath, run_id=run_id)

    data = deepcopy(data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if dataclasses.is_dataclass(data):
        data = dataclasses.asdict(data)

    if isinstance(data, list):
        for i in range(len(data)):
            if dataclasses.is_dataclass(data[i]):
                data[i] = dataclasses.asdict(data[i])

    # Add metadata for tracking (only for decoupled trajectory/belief files)
    if include_metadata and any(
        keyword in os.path.basename(filepath)
        for keyword in ["reasoning-trajectories-raw", "reasoning-beliefs"]
    ):
        metadata = create_file_metadata(
            {
                "file_type": (
                    "raw_trajectories" if "raw" in filepath else "belief_measurements"
                ),
                "run_id": run_id,
                "filename": os.path.basename(filepath),
            }
        )

        # Wrap data in new two-field structure with explicit ordering (metadata first)
        from collections import OrderedDict

        ordered_data = OrderedDict()
        ordered_data["metadata"] = metadata
        ordered_data["saved_content"] = data
        data = ordered_data

    if write_mode == "a" and not os.path.exists(filepath):
        write_mode = "w"

    with open(filepath, write_mode, encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        if write_mode == "a":
            f.write("\n")


def extract_json_from_str(s: str) -> Any:
    """
    Robustly extract JSON object from a string, after a wide range of sanitization operations.

    :param s: The string to extract JSON object from. It could be, for example, generation by an LLM.
    :type s: str

    :return: The extracted JSON object. None upon failure.
    """
    # Strip leading/trailing whitespace and formatting characters (```, ```json, etc.)
    s = s.replace("```json", "```")

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(s.strip("`"))
    except json.JSONDecodeError:
        pass

    if "```" in s:
        if s.startswith("```") and s.endswith("```"):
            s = s[3:-3]
        elif s.count("```") != 2:
            return None
        else:
            s = s.split("```")[1]

    s = s.strip()

    if not s:
        return None

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"Failed to extract JSON from string: {e}")
        return None


def compute_hash(data: Any, length: int = 12) -> str:
    """
    Compute a hash of the given data for unique identification.

    :param data: Data to hash (will be JSON serialized if not string)
    :param length: Length of the hash string to return (default 12)
    :return: Hex hash string
    """
    import hashlib

    if isinstance(data, str):
        content = data
    else:
        content = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)

    return hashlib.sha256(content.encode()).hexdigest()[:length]
