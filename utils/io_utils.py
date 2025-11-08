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
            self.filename = (
                f"data/tmp/logs/console-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{uuid.uuid4()}.log"
            )
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
            self.file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message_notrunc}\n")

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
            and self.message_timestamps[message][0] < current_time - self.message_window_size_secs[message]
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
                self.message_stem_per_window_max_count.get(message_stem) or max_count or 1e10
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
    if any(objs[i].__getattribute__(attr) != objs[0].__getattribute__(attr) for i in range(len(objs))):
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
    return max(_shared_prefix_len(sa[i:], sb[j:]) for i in range(len(sa)) for j in range(len(sb)))


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
    metadata = {"timestamp": datetime.datetime.now().isoformat(), "environment_variables": get_documented_env_vars()}

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
        raise ValueError(f"Invalid run_id format or new format without algorithm name: {run_id}")


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
    run_ids = [f.name.split("run-")[1] for f in os.scandir(dir) if f.is_dir() and "run-" in f.name]

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
            assert len(path_parts) == 2, f"Expected exactly one run directory in {example_path}, got {path_parts}"
            runs_subdir = path_parts[0]
            assert os.path.exists(runs_subdir), f"Run directory {runs_subdir} does not exist"
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


def load_file(filepath: str, schema: type | None = None, none_on_error: bool = False, **kwargs) -> Any:
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
                data = [dacite_from_dict(data_class=schema, data=item, config=config) for item in data]
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


def load_file_for_run(filepath: str, run_id: str, schema: type | None = None, example_path: str = None) -> Any:
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
        keyword in os.path.basename(filepath) for keyword in ["reasoning-trajectories-raw", "reasoning-beliefs"]
    ):
        metadata = create_file_metadata(
            {
                "file_type": "raw_trajectories" if "raw" in filepath else "belief_measurements",
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


def get_trajectory_filename(
    algo_name: str,
    file_type: Literal["raw", "belief"] = "belief",
    judge_policy_colloquial_name: str | list[str] = None,
    run_id: str = None,
    requires_existence=False,
    example_path: str = None,
) -> str:
    """Get the appropriate trajectory filename based on decoupling mode."""
    from core.reasoning.schema import get_decouple_config  # Import here to avoid circular dependency

    algo_belief_equivalence_classes = {
        "MartingaleStrategy": "base",
        "GroundTruthAccuracy": "base",
    }

    if isinstance(judge_policy_colloquial_name, list):
        judge_policy_colloquial_name = "_".join(judge_policy_colloquial_name)

    config = get_decouple_config()

    if config["decouple_enabled"]:
        if file_type == "raw":
            return "reasoning-trajectories-raw.json"
        else:
            candidate_algo_names = [algo_name]
            if algo_name in algo_belief_equivalence_classes:
                class_name = algo_belief_equivalence_classes[algo_name]
                candidate_algo_names += [k for k, v in algo_belief_equivalence_classes.items() if v == class_name]

            for candidate_algo_name in candidate_algo_names:
                # Include judge policy colloquial name if provided for algorithms that have judge_policies
                if judge_policy_colloquial_name:
                    filename = f"reasoning-beliefs-{candidate_algo_name}-{judge_policy_colloquial_name}.json"
                else:
                    filename = f"reasoning-beliefs-{candidate_algo_name}.json"

                if (
                    run_id
                    and requires_existence
                    and not os.path.exists(complete_path(filename, run_id, example_path=example_path))
                ):
                    continue

                return filename

            return None
    else:
        return "reasoning-trajectories.json"


def load_trajectories_with_backward_compatibility(
    run_id: str,
    algo_name: str,
    prefer_raw: bool = True,
    judge_policy_colloquial_name: Optional[str | list[str]] = None,
    example_path: str = None,
) -> tuple[list["RawReasoningTrajectory"], list["ReasoningTrajectory"] | None]:  # noqa: F821
    """
    Load trajectories with full backward compatibility support.

    Args:
        run_id: Run ID
        algo_name: Algorithm name
        prefer_raw: Whether to prefer raw trajectories
        judge_policy_colloquial_name: Judge policy colloquial name
        example_path: When DIR_NAME does not directly contain runs as subdirectories, this path may be provided to infer the correct run directory
          - We take arbitrary prefixes of the path to use in place of DIR_NAME.

    Returns:
        tuple: (raw_trajectories: list[RawReasoningTrajectory],
                full_trajectories: list[ReasoningTrajectory] | None)

        When prefer_raw=True, tries to load raw trajectories first.
        When prefer_raw=False or raw trajectories don't exist, loads legacy format.

        For legacy files, extracts raw trajectories and returns (raw_trajectories, full_trajectories).
        For decoupled files, loads raw and belief files separately and returns (raw_trajectories, None) if beliefs missing.
    """
    from core.reasoning.schema import (  # Import here to avoid circular dependency
        RawReasoningTrajectory,
        ReasoningTrajectory,
        get_decouple_config,
    )

    config = get_decouple_config()

    # Try to load raw trajectories first if decoupling is enabled and prefer_raw is True
    if config["decouple_enabled"] and prefer_raw:
        raw_filename = get_trajectory_filename(
            algo_name,
            file_type="raw",
            judge_policy_colloquial_name=judge_policy_colloquial_name,
            example_path=example_path,
        )
        raw_path = complete_path(raw_filename, run_id, example_path=example_path)

        if os.path.exists(raw_path):
            raw_trajectories = load_file_for_run(
                raw_filename, run_id, schema=RawReasoningTrajectory, example_path=example_path
            )

            # Try to load beliefs file
            beliefs_filename = get_trajectory_filename(
                algo_name,
                file_type="belief",
                judge_policy_colloquial_name=judge_policy_colloquial_name,
                run_id=run_id,
                requires_existence=True,
                example_path=example_path,
            )
            beliefs_path = complete_path(beliefs_filename, run_id, example_path=example_path)

            # Fallback to belief filename without judge policy name for backward compatibility
            if (beliefs_path is None or not os.path.exists(beliefs_path)) and judge_policy_colloquial_name:
                fallback_beliefs_filename = get_trajectory_filename(
                    algo_name,
                    file_type="belief",
                    judge_policy_colloquial_name=None,
                    run_id=run_id,
                    requires_existence=True,
                    example_path=example_path,
                )
                fallback_beliefs_path = complete_path(fallback_beliefs_filename, run_id, example_path=example_path)
                if fallback_beliefs_path is not None and os.path.exists(fallback_beliefs_path):
                    beliefs_filename = fallback_beliefs_filename
                    beliefs_path = fallback_beliefs_path

            if beliefs_path is not None and os.path.exists(beliefs_path):
                # Load belief data and combine with raw trajectories
                belief_data = load_file_for_run(beliefs_filename, run_id, example_path=example_path)
                full_trajectories = []

                for i, raw_traj in enumerate(raw_trajectories):
                    if i < len(belief_data):
                        beliefs = belief_data[i].get("beliefs", [None] * len(raw_traj.steps))
                        full_trajectories.append(raw_traj.to_reasoning_trajectory(beliefs))
                    else:
                        full_trajectories.append(raw_traj.to_reasoning_trajectory())

                return raw_trajectories, full_trajectories
            else:
                return raw_trajectories, None

    # Fall back to legacy trajectory loading
    legacy_filename = "reasoning-trajectories.json"
    legacy_path = complete_path(legacy_filename, run_id, example_path=example_path)

    if os.path.exists(legacy_path):
        full_trajectories = load_file_for_run(
            legacy_filename, run_id, schema=ReasoningTrajectory, example_path=example_path
        )
        raw_trajectories = [traj.to_raw_trajectory() for traj in full_trajectories]
        return raw_trajectories, full_trajectories

    # If no files exist, raise FileNotFoundError
    print(f"Prefer raw: {prefer_raw}; Config: {config}")
    if config["decouple_enabled"] and prefer_raw:
        print(f"Raw filename & path: {raw_filename} {raw_path} (exists: {os.path.exists(raw_path)})")
    print(f"Legacy filename & path: {legacy_filename} {legacy_path} (exists: {os.path.exists(legacy_path)})")
    raise FileNotFoundError(f"No trajectory files found for run {run_id}")


def save_trajectories_with_decoupling(
    trajectories: list,
    run_id: str,
    algo_name: str,
    beliefs_only: bool = False,
    judge_policy_colloquial_name: str = None,
) -> None:
    """
    Save trajectories respecting decoupling configuration.

    Args:
        trajectories: List of ReasoningTrajectory or RawReasoningTrajectory objects
        run_id: Run identifier
        algo_name: Algorithm name for belief files
        beliefs_only: If True, only save belief measurements (assumes raw trajectories exist)
    """
    from core.reasoning.schema import (  # Import here to avoid circular dependency
        RawReasoningTrajectory,
        ReasoningTrajectory,
        get_decouple_config,
    )

    config = get_decouple_config()

    if not config["decouple_enabled"]:
        # Legacy mode - save everything in one file
        if not beliefs_only:
            dump_file_for_run("reasoning-trajectories.json", trajectories, run_id)
        return

    # Decoupled mode
    if beliefs_only:
        # Save only belief measurements
        belief_data = []
        for traj in trajectories:
            if isinstance(traj, ReasoningTrajectory):
                belief_data.append(
                    {
                        "problem_id": traj.problem.id if hasattr(traj.problem, "id") else None,
                        "beliefs": traj.extract_beliefs(),
                    }
                )
            else:
                raise ValueError("beliefs_only=True requires ReasoningTrajectory objects")

        beliefs_filename = get_trajectory_filename(
            algo_name, file_type="belief", judge_policy_colloquial_name=judge_policy_colloquial_name, run_id=run_id
        )
        dump_file_for_run(beliefs_filename, belief_data, run_id)
    else:
        # Save raw trajectories and belief measurements separately
        if isinstance(trajectories[0], ReasoningTrajectory):
            # Extract raw trajectories
            raw_trajectories = [traj.to_raw_trajectory() for traj in trajectories]

            # Save raw trajectories
            raw_filename = get_trajectory_filename(
                algo_name, file_type="raw", judge_policy_colloquial_name=judge_policy_colloquial_name
            )
            dump_file_for_run(raw_filename, raw_trajectories, run_id)

            # Save belief measurements
            belief_data = []
            for traj in trajectories:
                belief_data.append(
                    {
                        "problem_id": traj.problem.id if hasattr(traj.problem, "id") else None,
                        "beliefs": traj.extract_beliefs(),
                    }
                )

            beliefs_filename = get_trajectory_filename(
                algo_name, file_type="belief", judge_policy_colloquial_name=judge_policy_colloquial_name, run_id=run_id
            )
            dump_file_for_run(beliefs_filename, belief_data, run_id)
        else:
            # Already raw trajectories, just save them
            raw_filename = get_trajectory_filename(
                algo_name, file_type="raw", judge_policy_colloquial_name=judge_policy_colloquial_name
            )
            dump_file_for_run(raw_filename, trajectories, run_id)


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


def extract_per_trajectory_scores_multi(
    bias_file_paths: list[str], trajectory_file_paths: list[str], check_alignment: bool = False
) -> list[list[tuple]]:
    """
    Extract per-trajectory scores from multiple bias evaluation results files.

    :param bias_file_paths: List of paths to bias-eval-results-*.json files
    :param trajectory_file_paths: List of paths to corresponding trajectory files
    :param check_alignment: If True, verify that question statements are identical across models
    :return: List of lists - per-trajectory scores for each model
    :raises ValueError: If check_alignment=True and questions don't match
    """
    if len(bias_file_paths) != len(trajectory_file_paths):
        raise ValueError(
            f"Number of bias files ({len(bias_file_paths)}) must match trajectory files ({len(trajectory_file_paths)})"
        )

    all_results = []
    for bias_path, traj_path in zip(bias_file_paths, trajectory_file_paths, strict=False):
        trajectory_score_pairs = extract_per_trajectory_scores(bias_path, traj_path)
        all_results.append(trajectory_score_pairs)

    if check_alignment and len(all_results) > 1:
        # Extract question statements from each model's trajectories
        all_questions = []
        for results in all_results:
            questions = []
            for traj, _ in results:
                if hasattr(traj, "problem") and hasattr(traj.problem, "question_statement"):
                    questions.append(traj.problem.question_statement)
                else:
                    # Handle dict format
                    if isinstance(traj, dict) and "problem" in traj:
                        q = traj["problem"].get("question_statement", "")
                        questions.append(q)
            all_questions.append(questions)

        # Find minimum length
        min_len = min(len(q) for q in all_questions)

        # Truncate to minimum length
        all_questions = [q[:min_len] for q in all_questions]
        all_results = [r[:min_len] for r in all_results]

        # Check alignment
        reference_questions = all_questions[0]
        for i, questions in enumerate(all_questions[1:], 1):
            if questions != reference_questions:
                # Find first mismatch for error message
                for j, (q1, q2) in enumerate(zip(reference_questions, questions, strict=False)):
                    if q1 != q2:
                        raise ValueError(
                            f"Question mismatch at index {j} between model 0 and model {i}:\n"
                            f"Model 0: {q1[:100]}...\n"
                            f"Model {i}: {q2[:100]}..."
                        )

    return all_results


def extract_per_trajectory_scores(bias_file_path: str, trajectory_file_path: str) -> list[tuple]:
    """
    Extract per-trajectory scores from a bias evaluation results file and pair them with trajectories.

    :param bias_file_path: Path to the bias-eval-results-*.json file
    :param trajectory_file_path: Path to the corresponding trajectory file
    :return: List of (trajectory, score) tuples sorted by score (ascending)
    :raises ValueError: If the DebiasStrategy doesn't support per-trajectory scores
    """
    from core.reasoning.schema import RawReasoningTrajectory, ReasoningTrajectory

    # Load bias results
    bias_results = load_file(bias_file_path)
    if not bias_results:
        raise FileNotFoundError(f"Could not load bias results from {bias_file_path}")

    # Extract strategy name from filename
    filename = os.path.basename(bias_file_path)
    if not filename.startswith("bias-eval-results-"):
        raise ValueError(f"Invalid bias results filename: {filename}")

    strategy_part = filename[len("bias-eval-results-") : -len(".json")]
    # Remove any suffix after underscore (e.g., MutualPredictStrategy_Qwen3-0 -> MutualPredictStrategy)
    base_strategy = strategy_part.split("_")[0].split("-")[0]

    # Load trajectories
    path_parts = bias_file_path.split("/")
    run_dir_names = [p for p in path_parts if p.startswith("run-")]
    assert len(run_dir_names) == 1, f"Expected exactly one run directory in {bias_file_path}, got {run_dir_names}"
    run_dir_name = run_dir_names[0]

    run_id = run_dir_name.removeprefix("run-")
    algo_name = bias_results["algo"]

    # Try to extract judge policy names, but don't fail if it can't be imported
    try:
        from utils.policy_utils import extract_policy_names_from_path

        judge_policy_colloquial_names = extract_policy_names_from_path(path_parts[-1])
    except (ImportError, Exception):
        judge_policy_colloquial_names = None

    raw_trajectories, trajectories = load_trajectories_with_backward_compatibility(
        run_id,
        algo_name,
        example_path=bias_file_path,
        judge_policy_colloquial_name=judge_policy_colloquial_names or None,
    )
    if not trajectories and base_strategy == "GroundTruthAccuracy":  # GTA requires beliefs
        raise FileNotFoundError(f"Could not load trajectories from {trajectory_file_path}")
    if not raw_trajectories:
        raise FileNotFoundError(f"Could not load raw trajectories from {trajectory_file_path}")

    # Convert raw trajectories to ReasoningTrajectory if needed
    if raw_trajectories and not trajectories:
        trajectories = [raw_traj.to_reasoning_trajectory() for raw_traj in raw_trajectories]

    if trajectories and isinstance(trajectories[0], dict) and "steps" in trajectories[0]:
        # Check if it's a raw trajectory
        if (
            "problem" in trajectories[0]
            and isinstance(trajectories[0]["steps"][0], dict)
            and "belief" not in trajectories[0]["steps"][0]
        ):
            # Raw trajectory format
            trajectories = [
                dacite_from_dict(RawReasoningTrajectory, t, config=DaciteConfig(check_types=False))
                for t in trajectories
            ]
            trajectories = [t.to_reasoning_trajectory() for t in trajectories]
        else:
            # Regular trajectory format
            trajectories = [
                dacite_from_dict(ReasoningTrajectory, t, config=DaciteConfig(check_types=False)) for t in trajectories
            ]

    # Extract per-trajectory scores using the strategy's parse method
    loss_details = bias_results.get("loss_details", {})

    # Try to get the strategy class and use its methods
    try:
        # Import all strategy classes
        from core.algo.accuracy import GroundTruthAccuracy
        from core.algo.graderwrapper import GraderWrapper
        from core.algo.mutualpredict import MutualPredictStrategy
        from core.algo.qualitative import QualitativeJudge
        from core.algo.worldintheloop import WorldInTheLoop

        # Map strategy names to classes
        strategy_classes = {
            "MutualPredictStrategy": MutualPredictStrategy,
            "QualitativeJudge": QualitativeJudge,
            "WorldInTheLoop": WorldInTheLoop,
            "GroundTruthAccuracy": GroundTruthAccuracy,
            "GraderWrapper": GraderWrapper,
        }

        strategy_class = strategy_classes.get(base_strategy)
        if strategy_class:
            # Create a dummy instance to call methods
            # We don't need real init params since we're just calling parse methods
            try:
                dummy_instance = strategy_class.__new__(strategy_class)
                if (
                    hasattr(dummy_instance, "supports_per_trajectory_scores")
                    and dummy_instance.supports_per_trajectory_scores()
                ):
                    scores = dummy_instance.parse_per_trajectory_scores(loss_details, trajectories)
                else:
                    raise ValueError(f"Strategy '{base_strategy}' does not support per-trajectory scores")
            except TypeError as e:
                # If we can't create instance, fall back to old logic
                raise ValueError(f"Could not create instance of '{base_strategy}' to extract scores") from e
        else:
            raise ValueError(
                f"Unknown strategy '{base_strategy}'. Supported strategies: {', '.join(strategy_classes.keys())}"
            )

    except (ImportError, AttributeError, ValueError) as e:
        # Fall back to legacy extraction logic if imports fail or methods don't exist
        print(f"Note: Using legacy extraction logic for {base_strategy}: {e}")

        supported_strategies = [
            "MutualPredictStrategy",
            "QualitativeJudge",
            "WorldInTheLoop",
            "GroundTruthAccuracy",
            "GraderWrapper",
        ]
        if base_strategy not in supported_strategies:
            raise ValueError(
                f"Strategy '{base_strategy}' does not provide per-trajectory scores or scores could not be extracted. "
                f"Supported strategies: {', '.join(supported_strategies)}"
            ) from e

        scores = None

        if base_strategy == "MutualPredictStrategy":
            # MutualPredictStrategy stores scores in all_logprobs_original_order
            if "all_logprobs_original_order" in loss_details:
                scores = loss_details["all_logprobs_original_order"]
            elif "per_response_details" in loss_details:
                # Alternative field for some versions
                scores = [
                    detail.get("uplift", detail.get("logprob", 0)) for detail in loss_details["per_response_details"]
                ]

        elif base_strategy == "QualitativeJudge":
            # QualitativeJudge stores scores in individual_scores
            if "individual_scores" in loss_details:
                scores = loss_details["individual_scores"]

        elif base_strategy == "WorldInTheLoop":
            # WorldInTheLoop stores scores in uplift_values
            if "uplift_values" in loss_details:
                scores = loss_details["uplift_values"]
            elif "proposition_results" in loss_details:
                # Try to extract from proposition_results if uplift_values not present
                prop_results = loss_details["proposition_results"]
                scores = []
                for result in prop_results:
                    if isinstance(result, dict):
                        # Look for uplift or reward fields
                        score = result.get("uplift", result.get("reward", result.get("score", None)))
                        if score is not None:
                            scores.append(score)

        elif base_strategy == "GroundTruthAccuracy":
            # GroundTruthAccuracy computes per-trajectory scores from beliefs
            if trajectories and all(
                hasattr(t, "problem") and hasattr(t.problem, "correct_option") for t in trajectories
            ):
                # Compute brier scores for each trajectory
                scores = []
                metric = loss_details.get("metric_name", "brier")
                mode = loss_details.get("mode_name", "eventual")

                for traj in trajectories:
                    if traj.problem.correct_option is not None:
                        # Get the relevant belief based on mode
                        if mode == "initial":
                            belief = traj.steps[0].belief if traj.steps else 0.5
                        elif mode == "eventual":
                            belief = traj.steps[-1].belief if traj.steps else 0.5
                        else:  # difference
                            initial = traj.steps[0].belief if traj.steps else 0.5
                            eventual = traj.steps[-1].belief if traj.steps else 0.5
                            belief = eventual - initial

                        # Compute score based on metric
                        correct = 1 - traj.problem.correct_option
                        if metric == "brier":
                            score = (correct - belief) ** 2
                        else:  # cross_entropy
                            import math

                            score = -correct * math.log(belief + 1e-18) - (1 - correct) * math.log(1 - belief + 1e-18)

                        scores.append(score)
                    else:
                        scores.append(None)  # No ground truth available

        elif base_strategy == "GraderWrapper":
            # GraderWrapper stores scores in per_trajectory_scores
            if "per_trajectory_scores" in loss_details:
                scores = loss_details["per_trajectory_scores"]

        # Check if we found scores
        if not scores:
            print(
                f"WARNING: No trajectory-score pairs found for {bias_file_path} and {trajectory_file_path} (loss_details keys: {loss_details.keys()})"
            )
            return []

    # Verify length match
    if len(scores) != len(trajectories):
        print(f"WARNING: Number of scores ({len(scores)}) doesn't match number of trajectories ({len(trajectories)})")
        # Use the minimum length
        min_len = min(len(scores), len(trajectories))
        scores = scores[:min_len]
        trajectories = trajectories[:min_len]

    # Pair trajectories with scores and filter out None scores
    trajectory_score_pairs = [
        (traj, score) for traj, score in zip(trajectories, scores, strict=False) if score is not None
    ]

    # Sort by score (ascending - lower is better for most metrics)
    trajectory_score_pairs.sort(key=lambda x: x[1])

    return trajectory_score_pairs


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
