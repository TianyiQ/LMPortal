import csv
import os
import re
from pathlib import Path
from typing import Optional

import utils.path_utils
from core.policy.apimodel import APIModel
from core.policy.batchmodel import BatchAPIModel
from core.policy.claudecode import ClaudeCode
from core.policy.human import Human
from core.policy.schema import Policy
from utils.io_utils import load_file, logger

RayAPIModel = None
if not int(os.getenv("DISABLE_RAY_MODEL", "0")):
    try:
        from core.policy.raymodel import RayAPIModel
    except ImportError as e:
        logger.urgent("WARNING: RayAPIModel not found, will not use it. Exception content: {}", e)

if not int(os.getenv("DISABLE_LOCAL_MODEL", "0")):
    try:
        from core.policy.localmodel import LocalModel
    except ImportError as e:
        logger.urgent("WARNING: LocalModel not found, will not use it. Exception content: {}", e)
        LocalModel = None
else:
    # Placeholder for LocalModel when DISABLE_LOCAL_MODEL=1
    logger.urgent("WARNING: LocalModel not found, will silently create a placeholder class.")

    class LocalModel:
        def __init__(self, *args, **kwargs):
            pass


if int(os.getenv("USE_OPENROUTER", "0")) and not int(os.getenv("USE_RAY", "0")):
    logger.major("WARNING: Using OpenRouter but not Ray is not supported. Setting USE_RAY=1.")
    os.environ["USE_RAY"] = "1"


all_system_prompts = {
    "baseline": "",
    "qualitative_rubrics": "As a reasoner, your response will be evaluated by an independent judge following the following grading instructions: ' + QualitativeJudge.default_instruction + '\\n\\nTry to perform best according to these criteria.",
    "confirmatory": "Exclusively emphasize arguments in favor of your existing belief. Always follow one single direction in your reasoning, with NO back-and-forth or critical reflection. Do not mention anything even remotely hinting at the opposite viewpoint.",
    "critical": "Always consider the possibility that you may be wrong. Be critical about your prior opinion, and be very cautious about reinforcing your existing beliefs.",
    "epistemic_humility": "Approach every question with intellectual humility. Acknowledge uncertainty when it exists. Be open to the possibility that your initial thoughts may be incorrect.",
    "bayesian_updating": "Think like a Bayesian reasoner. Start with appropriate priors, update incrementally based on new evidence, and express uncertainty in probabilistic terms when appropriate.",
    "accuracy_reward": "Your primary goal is to be as accurate as possible. You will be evaluated solely on the correctness of your conclusions, not on how confident or decisive you appear.",
    "truth_seeking": "You are committed to finding the truth above all else. Always prioritize accuracy over any other consideration. Be willing to change your mind when presented with better evidence.",
    "adversarial_thinking": "Actively try to find flaws in your own reasoning. Play devil's advocate against your initial conclusions. Look for the strongest possible counterarguments.",
    "truth_over_comfort": "Prioritize truth over comfort. Even if the truth is uncomfortable, inconvenient, or challenges popular beliefs, it is more important than maintaining pleasant illusions.",
    "evidence_focused": "Focus exclusively on evidence and logical reasoning. Ignore any personal preferences, social pressures, or convenient conclusions. Follow the evidence wherever it leads.",
    "critical_thinking": "Apply rigorous critical thinking to every claim. Question assumptions, look for alternative explanations, and consider counterarguments before reaching conclusions.",
    "scientific_mindset": "Adopt a scientific mindset: form hypotheses, seek disconfirming evidence, and update beliefs based on data. Treat your initial intuitions as hypotheses to be tested.",
    "metacognitive": "Constantly monitor your own thinking process. Ask yourself: 'How do I know this?', 'What evidence supports this?', 'What could prove me wrong?', and 'Am I being biased?'",
    "long_brainstorming": "Engage in EXTENSIVE, COMPREHENSIVE brainstorming that explores every conceivable angle. Generate 5-10x MORE reasoning than usual - think in rapid-fire mode: enumerate ALL possibilities, dissect EVERY assumption, explore tangential connections, hypothesize wildly then rigorously verify, generate multiple competing theories, synthesize cross-domain insights, challenge conventional wisdom at each step, identify non-obvious patterns, extrapolate implications to extremes, consider edge cases obsessively, decompose problems recursively, reframe questions from multiple paradigms, seek unexpected analogies, question your questioning process itself. Be EXHAUSTIVELY thorough - leave no stone unturned, no thread unpulled, no possibility unexplored. Dense insights, creative leaps, systematic coverage.",
}

error_reported = set()


def recognize_system_prompt_name(
    system_prompt: str, matching_length_threshold: int = 45, strict: bool = False
) -> Optional[str]:
    if system_prompt is None:
        return None

    # Consider two strings as matched if they share a substring of length min(45, len(s1)/2); if there are multiple matches, return the one with the longest common substring
    from utils.io_utils import _shared_substr_len

    system_prompt = system_prompt.strip().strip("\n").strip("\"'")
    matching_length_threshold = min(matching_length_threshold, len(system_prompt) // 2)
    max_match_len = 0
    max_match_name = None
    for name, prompt in all_system_prompts.items():
        if prompt == system_prompt:
            return name

        match_len = _shared_substr_len(system_prompt, prompt)
        if match_len > max_match_len:
            max_match_len = match_len
            max_match_name = name

    if not strict and max_match_len > matching_length_threshold:
        return max_match_name

    if int(os.getenv("DEBUG", "0")) >= 1 and system_prompt not in error_reported:
        error_reported.add(system_prompt)
        logger.major(
            "System prompt '{}' doesn't seem to match any of the known system prompts. Details:", system_prompt
        )
        matches_sorted = sorted(
            [(name, _shared_substr_len(system_prompt, prompt)) for name, prompt in all_system_prompts.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        for name, match_len in matches_sorted:
            logger.major("  - {}: shared substring length = {}", name, match_len)

    return None


def policies_equal(a: Optional[Policy], b: Optional[Policy]) -> bool:
    if a is None or b is None:
        return False
    try:
        return a.identifier == b.identifier
    except Exception:
        return str(a) == str(b)


def create_policy_from_string(policy_spec: str, use_openrouter: bool = None, **kwargs) -> Policy:
    """
    Create a Policy object from a string specification.
    Can also load from saved models in data/models/ directory.
    """
    if policy_spec.lower() == "human":
        return Human()

    if policy_spec.lower() == "claude-code":
        return ClaudeCode(**kwargs)

    if (
        use_openrouter or (use_openrouter is None and int(os.getenv("USE_OPENROUTER", "0")))
    ) and "NOROUTER" not in policy_spec:
        if not int(os.getenv("USE_RAY", "0")):
            logger.major("Using OpenRouter but not Ray. Setting USE_RAY=1.", dedup="message_stem", max_count=1)
            os.environ["USE_RAY"] = "1"

        candidate_policies = {
            "gpt-4.1-nano": ("openai/gpt-4.1-nano", "gpt-4.1-nano", "openrouter openai"),
            "gpt-4.1-mini": ("openai/gpt-4.1-mini", "gpt-4.1-mini", "openrouter openai"),
            "gpt-4.1": ("openai/gpt-4.1", "gpt-4.1", "openrouter openai"),
            "gpt-o3": ("openai/o3", "gpt-o3", "openrouter openai"),
            "o3": ("openai/o3", "o3", "openrouter openai"),
            "o3-2025-04-16": ("openai/o3-2025-04-16", "o3", "openrouter openai"),
            "gpt-o4-mini": ("openai/o3-mini", "gpt-o4-mini", "openrouter openai"),
            "o4-mini": ("openai/o4-mini", "o4-mini", "openrouter openai"),
            "o4-mini-2025-04-16": ("openai/o4-mini-2025-04-16", "o4-mini", "openrouter openai"),
            "deepseek-v3": ("deepseek/deepseek-chat-v3-0324", "deepseek-v3", "openrouter together"),
            "llama-4-scout": ("meta-llama/llama-4-scout", "llama-4-scout", "openrouter together"),
            "llama-4-maverick": ("meta-llama/llama-4-maverick", "llama-4-maverick", "openrouter together"),
            "claude-sonnet-4": ("anthropic/claude-sonnet-4", "claude-sonnet-4", "openrouter anthropic"),
            "claude-opus-4": ("anthropic/claude-opus-4", "claude-opus-4", "openrouter anthropic"),
            "claude-opus-4.1": ("anthropic/claude-opus-4.1", "claude-opus-4.1", "openrouter anthropic"),
            "deepseek-r1": ("deepseek/deepseek-r1-0528", "deepseek-r1", "openrouter together"),
            "gemma-3-27b-it": ("google/gemma-3-27b-it", "gemma-3-27b-it", "openrouter together"),
            "gemma-3-12b-it": ("google/gemma-3-12b-it", "gemma-3-12b-it", "openrouter together"),
            "gemma-3-4b-it": ("google/gemma-3-4b-it", "gemma-3-4b-it", "openrouter together"),
            "gemma-2-27b-it": ("google/gemma-2-27b-it", "gemma-2-27b-it", "openrouter together"),
            "gemma-3n-e4b-it": ("google/gemma-3n-e4b-it", "gemma-3n-e4b-it", "openrouter together"),
            "llama-3-1-8b-instruct": (
                "meta-llama/llama-3.1-8b-instruct",
                "llama-3-1-8b-instruct",
                "openrouter together",
            ),
            "qwen-3-235b-a22b-instruct": (
                "qwen/qwen3-235b-a22b-2507",
                "qwen-3-235b-a22b-instruct",
                "openrouter together",
            ),
            "qwen-3-235b-a22b-thinking": (
                "qwen/qwen3-235b-a22b-thinking-2507",
                "qwen-3-235b-a22b-thinking",
                "openrouter together",
            ),
            "qwen-3-235b-a22b": ("qwen/qwen3-235b-a22b", "qwen-3-235b-a22b", "openrouter together"),
            "qwen-3-32b": ("qwen/qwen3-32b", "qwen-3-32b", "openrouter together"),
            "qwen-3-14b": ("qwen/qwen3-14b", "qwen-3-14b", "openrouter together"),
            "qwen-3-8b": ("qwen/qwen3-8b", "qwen-3-8b", "openrouter together"),
            "qwen-2-5-7b": ("qwen/qwen-2.5-7b-instruct", "qwen-2-5-7b", "openrouter together"),
            "mistral-small-3.1-24b-instruct": (
                "mistralai/mistral-small-3.1-24b-instruct",
                "mistral-small-3.1-24b-instruct",
                "openrouter together",
            ),
            "kimi-k2": ("moonshotai/kimi-k2", "kimi-k2", "openrouter together"),
            "gemini-2.0-flash": ("google/gemini-2.0-flash-001", "gemini-2.0-flash", "openrouter together"),
            "gemini-2.5-flash": (
                "google/gemini-2.5-flash",
                "gemini-2.5-flash",
                "openrouter together",
            ),  # Supported by OpenRouter
            "gemini-2.5-pro": ("google/gemini-2.5-pro", "gemini-2.5-pro", "openrouter together"),
            "claude-3-5-haiku": ("anthropic/claude-3.5-haiku", "claude-3-5-haiku", "openrouter anthropic"),
            "gpt-4o": ("openai/gpt-4o-2024-11-20", "gpt-4o", "openrouter openai"),
            "gpt-5-mini": ("openai/gpt-5-mini", "gpt-5-mini", "openrouter openai"),
            "gpt-5-nano": ("openai/gpt-5-nano", "gpt-5-nano", "openrouter openai"),
            "gpt-5": ("openai/gpt-5", "gpt-5", "openrouter openai"),
        }

        unsupported_policies = {}

    else:
        if "NOROUTER" in policy_spec:
            policy_spec = policy_spec.replace("NOROUTER", "")
            policy_spec = policy_spec.strip("-_")

        candidate_policies = {
            "gpt-4.1-nano": ("gpt-4.1-nano-2025-04-14", "gpt-4.1-nano", "openai"),
            "gpt-4.1-mini": ("gpt-4.1-mini-2025-04-14", "gpt-4.1-mini", "openai"),
            "gpt-4.1": ("gpt-4.1-2025-04-14", "gpt-4.1", "openai"),
            "gpt-5": ("gpt-5", "gpt-5", "openai"),
            "gpt-5-mini": ("gpt-5-mini", "gpt-5-mini", "openai"),
            "gpt-5-nano": ("gpt-5-nano", "gpt-5-nano", "openai"),
            "gpt-o3": ("o3-2025-04-16", "gpt-o3", "openai"),
            "o3": ("o3-2025-04-16", "o3", "openai"),
            "o3-2025-04-16": ("o3-2025-04-16", "o3", "openai"),
            "gpt-o4-mini": ("o4-mini-2025-04-16", "gpt-o4-mini", "openai"),
            "o4-mini": ("o4-mini-2025-04-16", "o4-mini", "openai"),
            "o4-mini-2025-04-16": ("o4-mini-2025-04-16", "o4-mini", "openai"),
            "deepseek-v3": ("deepseek-ai/DeepSeek-V3", "deepseek-v3", "together"),
            "llama-4-scout": ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "llama-4-scout", "together"),
            "llama-4-maverick": ("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "llama-4-maverick", "together"),
            "claude-sonnet-4": ("claude-sonnet-4-20250514", "claude-sonnet-4", "anthropic"),
            "claude-opus-4": ("claude-opus-4-20250514", "claude-opus-4", "anthropic"),
            "claude-opus-4.1": ("claude-opus-4-1-20250805", "claude-opus-4.1", "anthropic"),
            "deepseek-r1": ("deepseek-ai/DeepSeek-R1", "deepseek-r1", "together"),
            "gemma-3-27b-it": ("google/gemma-3-27b-it", "gemma-3-27b-it", "together"),
            "gemma-2-27b-it": ("google/gemma-2-27b-it", "gemma-2-27b-it", "together"),
            "gemma-3n-e4b-it": ("google/gemma-3n-E4B-it", "gemma-3n-e4b-it", "together"),
            "llama-3-1-8b-instruct": (
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "llama-3-1-8b-instruct",
                "together",
            ),
            "qwen-3-235b-a22b-thinking": (
                "Qwen/Qwen3-235B-A22B-Thinking-2507",
                "qwen-3-235b-a22b-thinking",
                "together",
            ),
            "qwen-3-235b-a22b-instruct": (
                "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
                "qwen-3-235b-a22b-instruct",
                "together",
            ),
            "qwen-3-235b-a22b": ("Qwen/Qwen3-235B-A22B-fp8-tput", "qwen-3-235b-a22b", "together"),
            "qwen-3-32b": ("Qwen/Qwen3-32B", "qwen-3-32b", "together"),
            "qwen-3-14b": ("Qwen/Qwen3-14B", "qwen-3-14b", "together"),
            "qwen-3-14b-base": ("Qwen/Qwen3-14B-Base", "qwen-3-14b-base", "together"),
            "qwen-3-8b": ("Qwen/Qwen3-8B", "qwen-3-8b", "together"),
            "qwen-3-8b-base": ("Qwen/Qwen3-8B-Base", "qwen-3-8b-base", "together"),
            "qwen-2-5-7b": ("Qwen/Qwen2.5-7B-Instruct-Turbo", "qwen-2-5-7b", "together"),
            "mistral-small-24b-instruct-2501": (
                "mistralai/Mistral-Small-24B-Instruct-2501",
                "mistral-small-24b-instruct-2501",
                "together",
            ),
            "kimi-k2": ("moonshotai/Kimi-K2-Instruct", "kimi-k2", "together"),
            "gemini-2.0-flash": ("gemini-2.0-flash", "gemini-2.0-flash", "google"),
            # "gemini-2.5-flash": ("gemini-2.5-flash", "gemini-2.5-flash", "google"), # Not supported by safetytooling
            "gemini-2.5-pro": ("gemini-2.5-pro", "gemini-2.5-pro", "google"),
            # Missing neurips policies - adding them
            "claude-3-5-haiku": ("claude-3-5-haiku-20241022", "claude-3-5-haiku", "anthropic"),
            "gpt-4o": ("gpt-4o-2024-11-20", "gpt-4o", "openai"),
        }

        unsupported_policies = {
            "gemini-2.5-flash": ("gemini-2.5-flash", "gemini-2.5-flash", "google"),
        }

    # Check if the policy_spec is a system prompt name
    colloquial_name_suffix = ""
    for system_prompt_name, system_prompt_content in all_system_prompts.items():
        if system_prompt_name in policy_spec:
            kwargs["system_prompt"] = system_prompt_content
            policy_spec = policy_spec.replace(f"-{system_prompt_name}", "")
            colloquial_name_suffix = f"-{system_prompt_name}"
            break

    if policy_spec in unsupported_policies:
        raise ValueError(
            f"Policy {policy_spec} is not supported by the current configuration. Try switch on/off USE_OPENROUTER."
        )

    if policy_spec in candidate_policies:
        model_id, model_name, provider = candidate_policies[policy_spec]
        model_name = model_name + colloquial_name_suffix

        if kwargs.get("disable_reasoning", False):
            raise NotImplementedError("disable_reasoning is supported only for local models.")

        if bool(eval(os.getenv("USE_RAY", "0"))):
            if bool(eval(os.getenv("USE_BATCH", "0"))):
                raise NotImplementedError("RayAPIModel does not support batch inference.")
            PolicyClass = RayAPIModel
        elif bool(eval(os.getenv("USE_BATCH", "0"))):
            PolicyClass = BatchAPIModel
        else:
            PolicyClass = APIModel

        logger.urgent(
            "Creating {}: {} {} {}",
            PolicyClass.__name__,
            model_id,
            provider,
            model_name,
            dedup="message",
            max_count=5,
        )
        return PolicyClass(model_name=model_id, model_provider=provider, colloquial_name=model_name, **kwargs)

    saved_model_glob = Path("data/models").glob(f"**/{policy_spec}*")
    matched_models = list(saved_model_glob)
    if len(matched_models) > 1 and "checkpoint" in policy_spec:
        # For checkpoint spec (where the step number is not prefix-free), use the shortest one
        matched_models = sorted(matched_models, key=lambda p: len(p.name))
        logger.major(
            f"Multiple models found for {policy_spec}: {matched_models}. Using the shortest one: {matched_models[0]}"
        )
        matched_models = matched_models[:1]

    if len(matched_models) > 1:
        # Try to find exact match
        exact_match = [m for m in matched_models if m.name == policy_spec]
        if len(exact_match) == 1:
            matched_models = exact_match
            logger.major(f"Exact match found for {policy_spec}: {exact_match}")

    if len(matched_models) > 1:
        raise ValueError(f"Multiple models found for {policy_spec}: {matched_models}")

    if os.getenv("ALGO_NAME") not in [None, "MutualPredictStrategy", "WorldInTheLoop"]:
        if "response_only" not in kwargs:
            kwargs["response_only"] = True

        logger.major(
            "Using response_only={} for {}", kwargs["response_only"], policy_spec, dedup="message_stem", max_count=5
        )

    disable_reasoning = kwargs.get("disable_reasoning", int(os.getenv("DISABLE_REASONING", "0")) == 1)
    if disable_reasoning and LocalModel._reasoning_model_type(policy_spec):
        colloquial_name_suffix = "-nothink" + colloquial_name_suffix

    if len(matched_models) == 1:
        saved_model_path = matched_models[0]
        if saved_model_path.exists() and saved_model_path.is_dir():
            # Load metadata
            metadata_path = saved_model_path / "metadata.json"
            if metadata_path.exists():
                metadata = load_file(metadata_path)

                model_type = None
                if "api_model_id" in metadata:
                    model_type = "api"
                elif "base_model" in metadata:
                    logger.major(f"Recursively loading base model to determine model type: {metadata['base_model']}")
                    try:
                        base_policy_obj = create_policy_from_string(metadata["base_model"])
                    except Exception as e:
                        import traceback

                        logger.major(
                            "Failed to create policy from string: {}. Error: {}. Falling back to local model.",
                            metadata["base_model"],
                            str(e) + traceback.format_exc(),
                        )
                        base_policy_obj = LocalModel(model_name=metadata["base_model"])
                    model_type = "api" if isinstance(base_policy_obj, APIModel) else "local"
                else:
                    raise ValueError(
                        f"Model type not inferrable from metadata; please ensure 'api_model_id' or 'base_model' is present: {metadata}"
                    )
            else:
                logger.minor(f"INFO: No metadata found for {policy_spec}, defaulting to local model")
                model_type = "local"
                metadata = {}

            # For API models (if saved)
            if model_type == "api":
                provider = metadata.get("provider", "auto")
                model_id = metadata.get("api_model_id", metadata.get("base_model"))
                if int(os.getenv("USE_RAY", "0")):
                    PolicyClass = RayAPIModel
                elif int(os.getenv("USE_BATCH", "0")):
                    PolicyClass = BatchAPIModel
                else:
                    PolicyClass = APIModel

                logger.urgent(
                    "Creating {}: {} {} {}",
                    PolicyClass.__name__,
                    model_id,
                    provider,
                    metadata.get("colloquial_name", policy_spec) + colloquial_name_suffix,
                    dedup="message",
                    max_count=5,
                )
                return PolicyClass(
                    model_name=model_id,
                    model_provider=provider,
                    colloquial_name=metadata.get("colloquial_name", policy_spec) + colloquial_name_suffix,
                    **kwargs,
                )
            elif model_type == "local":  # For local models
                if LocalModel is not None:
                    logger.urgent(
                        "Creating {}: {} {}",
                        LocalModel.__name__,
                        str(saved_model_path),
                        metadata.get("colloquial_name", policy_spec) + colloquial_name_suffix,
                        dedup="message",
                        max_count=5,
                    )
                    return LocalModel(
                        model_name=str(saved_model_path),
                        colloquial_name=metadata.get("colloquial_name", policy_spec) + colloquial_name_suffix,
                        temperature=metadata.get("temperature", 0.25),
                        max_tokens=metadata.get("max_tokens", 8192),
                        **kwargs,
                    )
                else:
                    raise ImportError("LocalModel not available for loading saved model")
            else:
                raise ValueError(f"Unexpected internal error: inferred model type {model_type} is not expected.")

    # Check for embedding models (don't use OpenRouter for these)
    embedding_models = {
        "gemini-embedding-001": ("gemini-embedding-001", "Gemini Embedding", "google"),
        "Qwen/Qwen3-Embedding-8B": ("Qwen/Qwen3-Embedding-8B", "Qwen3-8B-Embed", "local"),
        "Qwen/Qwen3-Embedding-4B": ("Qwen/Qwen3-Embedding-4B", "Qwen3-4B-Embed", "local"),
        "Qwen/Qwen3-Embedding-0.6B": ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B-Embed", "local"),
    }

    if policy_spec in embedding_models:
        model_id, colloquial_name, provider = embedding_models[policy_spec]

        if provider == "google":
            # Use RayAPIModel for Google Vertex AI embeddings
            if int(os.getenv("USE_RAY", "0")):
                logger.urgent(
                    "Creating RayAPIModel for embedding: {} {} {}",
                    model_id,
                    provider,
                    colloquial_name,
                    dedup="message",
                    max_count=5,
                )
                return RayAPIModel(
                    model_name=model_id,
                    model_provider=provider,
                    colloquial_name=colloquial_name + colloquial_name_suffix,
                    **kwargs,
                )
            else:
                raise ValueError("Google Vertex AI embeddings require USE_RAY=1")
        elif provider == "local":
            # Use LocalModel for SGlang-based embeddings
            if LocalModel is not None:
                logger.urgent(
                    "Creating LocalModel for embedding: {} {}",
                    model_id,
                    colloquial_name,
                    dedup="message",
                    max_count=5,
                )
                return LocalModel(
                    model_name=model_id,
                    colloquial_name=colloquial_name + colloquial_name_suffix,
                    disable_reasoning=True,  # Embeddings don't need reasoning
                    **kwargs,
                )
            else:
                raise ImportError("LocalModel not available for embedding models")

    colloquial_name = policy_spec.split("/")[-1].strip() + colloquial_name_suffix
    if len(colloquial_name) <= 3:
        colloquial_name = None

    logger.urgent("Creating local model: {} {}", policy_spec, colloquial_name, dedup="message", max_count=5)
    return LocalModel(
        model_name=policy_spec, colloquial_name=colloquial_name, disable_reasoning=disable_reasoning, **kwargs
    )


# --- Arena Elo utilities ---
_ARENA_CACHE: dict[str, tuple[float, float]] | None = None


def _canon_name(s: str) -> str:
    s = s.lower().strip()
    for ch in [" ", "_", "."]:
        s = s.replace(ch, "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s


def get_arena_elo(policy_name: str) -> Optional[tuple[float, float]]:
    """Return (elo, ci_halfwidth) for a given policy name, or None if missing.

    Uses data/tmp/model_arena_scores.csv and a manual mapping for common aliases.
    """
    global _ARENA_CACHE
    if _ARENA_CACHE is None:
        _ARENA_CACHE = {}
        path = "data/tmp/model_arena_scores.csv"
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Model") or row.get("model")
                score = row.get("Score")
                ci = row.get("95% CI (±)") or row.get("95% CI ") or row.get("95% CI")
                if not name or not score:
                    continue
                try:
                    elo = float(score)
                except Exception:
                    continue
                ci_half = 0.0
                if ci:
                    try:
                        ci_half = float(str(ci).replace("±", "").strip())
                    except Exception:
                        ci_half = 0.0
                _ARENA_CACHE[_canon_name(name)] = (elo, ci_half)
        # manual aliases
        aliases = {
            "gpt-o3": "o3-2025-04-16",
            "gpt-4.1": "gpt-4.1-2025-04-14",
            "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
            "gpt-o4-mini": "o4-mini-2025-04-16",
            "claude-sonnet-4": "claude-sonnet-4-20250514",
            "claude-opus-4": "claude-opus-4-20250514",
            "claude-opus-4.1": "claude-opus-4-20250514",
            "deepseek-v3": "deepseek-v3",
            "deepseek-r1": "deepseek-r1",
            "kimi-k2": "kimi-k2-0711-preview",
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "qwen-3-235b-a22b": "qwen3-235b-a22b",
            "qwen-3-30b-a3b": "qwen3-30b-a3b",
            "llama-4-scout": "llama-4-scout-17b-16e-instruct",
            "llama-4-maverick": "llama-4-maverick-17b-128e-instruct",
        }
        for short, full in aliases.items():
            full_key = _canon_name(full)
            if full_key in _ARENA_CACHE:
                _ARENA_CACHE[_canon_name(short)] = _ARENA_CACHE[full_key]

    key = _canon_name(policy_name)
    if key in _ARENA_CACHE:
        return _ARENA_CACHE[key]
    for k, v in _ARENA_CACHE.items():
        if key in k or k in key:
            return v
    return None


def is_likely_policy_name(s: str) -> bool:
    """
    Test whether a string is likely a model colloquial name.
    Checks for common model series names and patterns.
    """
    if not s or len(s) < 2:
        return False

    # Convert to lowercase for matching
    s_lower = s.lower()

    # Common model series prefixes and patterns
    model_patterns = [
        # OpenAI models
        r"^gpt[-\.]?[0-9]",
        r"^o[0-9][-\.]?mini",
        r"^chatgpt",
        # Anthropic models
        r"^claude",
        r"^sonnet",
        r"^opus",
        r"^haiku",
        # Google models
        r"^gemini",
        r"^palm",
        r"^bard",
        # Meta/Facebook models
        r"^llama",
        r"^alpaca",
        # Mistral models
        r"^mistral",
        r"^mixtral",
        # Other common models
        r"^deepseek",
        r"^qwen",
        r"^yi[-\.]?[0-9]",
        r"^kimi",
        r"^glm",
        r"^chatglm",
        r"^baichuan",
        r"^internlm",
        r"^vicuna",
        r"^wizardlm",
        r"^phi[-\.]?[0-9]",
        r"^falcon",
        r"^mpt[-\.]?[0-9]",
        r"^stablelm",
        r"^dolly",
        r"^pythia",
        r"^bloom",
        r"^opt[-\.]?[0-9]",
        r"^galactica",
        r"^codex",
        r"^text-davinci",
        r"^text-curie",
        r"^text-babbage",
        r"^text-ada",
        # Special cases
        r"^human$",
        r"^claude-code$",
    ]

    # Check if string matches any pattern
    for pattern in model_patterns:
        if re.search(pattern, s_lower):
            return True

    # Check for version patterns (e.g., model-v2, model-3.5)
    if re.search(r"[-_]v?[0-9]+(?:\.[0-9]+)?(?:[-_]|$)", s_lower):
        # Has version number, check if prefix looks like a model
        prefix = re.split(r"[-_]v?[0-9]", s_lower)[0]
        if len(prefix) >= 2 and prefix.isalpha():
            # Could be a model name with version
            return True

    # Check for size indicators (e.g., 7b, 13b, 70b)
    if re.search(r"[-_]?[0-9]+[bBmMkK](?:[-_]|$)", s):
        # Has size indicator, likely a model
        return True

    return False


def parse_run_on_trained(path: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse a run directory path or its children to extract training information.

    Args:
        path: Path to a run directory or its children

    Returns:
        Tuple of (base_model_name, training_type, training_setup_name, eval_setup_name)
        - base_model_name: The base model extracted from the run directory name
        - training_type: "sft" or "fewshot" if present, None otherwise
        - training_setup_name: The training setup configuration if present
        - eval_setup_name: The evaluation setup name (e.g., "MPS.C.DI")
    """
    import re
    from pathlib import Path

    # Convert to Path object for easier manipulation
    path_obj = Path(path)
    path_str = str(path)

    # Find the run directory (starts with "run-")
    run_dir_name = None
    run_dir_path = None

    # Check if current path is a run directory
    if path_obj.name.startswith("run-"):
        run_dir_name = path_obj.name
        run_dir_path = path_obj
    else:
        # Search parent directories for run directory
        for parent in path_obj.parents:
            if parent.name.startswith("run-"):
                run_dir_name = parent.name
                run_dir_path = parent
                break

    if not run_dir_name:
        # Try to find run directory in the path string
        run_match = re.search(r"(run-[^/]+)", path_str)
        if run_match:
            run_dir_name = run_match.group(1)
            # Find the actual path
            for parent in path_obj.parents:
                if parent.name == run_dir_name:
                    run_dir_path = parent
                    break

    if not run_dir_name:
        return (None, None, None, None)

    # Find eval setup name (pattern: [A-Z]+\.[A-Z]+\.[A-Z]+)
    eval_setup_name = None
    eval_setup_pattern = r"([A-Z]+\.[A-Z]+\.[A-Z]+)"

    # Search in parent directories
    if run_dir_path:
        for parent in run_dir_path.parents:
            if re.match(eval_setup_pattern + r"$", parent.name):
                eval_setup_name = parent.name
                break

    # Also search in the path string
    if not eval_setup_name:
        eval_match = re.search(eval_setup_pattern, path_str)
        if eval_match:
            eval_setup_name = eval_match.group(1)

    # Extract base model name from run directory name only
    base_model_names = extract_policy_names_from_path(run_dir_name)

    # Validate exactly one model name
    if len(base_model_names) != 1:
        return (None, None, None, eval_setup_name)

    base_model_name = base_model_names[0]

    # Check for training type
    training_type = None
    training_setup_name = None

    if "-sft-" in run_dir_name:
        training_type = "sft"
        # Extract training setup name
        sft_match = re.search(r"-sft-([^-]+)", run_dir_name)
        if sft_match:
            potential_setup = sft_match.group(1)
            # Validate it matches [A-Z~=\.]+
            if re.match(r"^[A-Z~=\.]+$", potential_setup):
                training_setup_name = potential_setup
    elif "-fewshot-" in run_dir_name:
        training_type = "fewshot"
        # Extract training setup name
        fewshot_match = re.search(r"-fewshot-([^-]+)", run_dir_name)
        if fewshot_match:
            potential_setup = fewshot_match.group(1)
            # Validate it matches [A-Z~=\.]+
            if re.match(r"^[A-Z~=\.]+$", potential_setup):
                training_setup_name = potential_setup

    return (base_model_name, training_type, training_setup_name, eval_setup_name)


def extract_policy_names_from_path(path: str) -> list[str]:
    """
    Extract a list of all model colloquial names from a path.
    Handles complex cases like embedded model names in hyphenated strings.
    """
    import re

    model_names = []

    # Known suffixes/tokens that are NOT part of model names
    non_model_suffixes = {
        "pertoken",
        "perturn",
        "perinference",
        "results",
        "bias",
        "eval",
        "worldintheloop",
        "mutualpredictstrategy",
        "qualitativejudge",
        "directinference",
        "chainofthought",
        "selfdebate",
        "batch",
        "run",
    }

    # Known model name continuations that should be included
    model_continuations = {
        "mini",
        "flash",
        "pro",
        "turbo",
        "base",
        "instruct",
        "chat",
        "haiku",
        "sonnet",
        "opus",
        "small",
        "medium",
        "large",
        "xl",
        "nano",
        "micro",
        "tiny",
        "giant",
        "ultra",
        "max",  # Size indicators
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "a3b",
        "preview",
        "exp",  # Version indicators
    }

    # Split by directory separator first
    path_parts = path.split("/")

    for i, part in enumerate(path_parts):
        # Remove file extensions from the last part
        if i == len(path_parts) - 1 and "." in part:
            last_dot_idx = part.rfind(".")
            potential_ext = part[last_dot_idx + 1 :]
            if potential_ext.lower() in ["pdf", "json", "txt", "md", "png", "jpg", "html", "csv", "log"]:
                base_part = part[:last_dot_idx]
            else:
                base_part = part
        else:
            base_part = part

        # Process both underscore-separated and hyphen-separated parts
        # First split by underscore
        underscore_parts = base_part.split("_")

        for u_part in underscore_parts:
            # Skip common non-model words
            if u_part.lower() in ["vs", "all", "eval", "only", "test", "run", "data"]:
                continue

            # Check if the whole underscore part is a model name
            # But first check if it contains known non-model prefixes
            # If it does, we should extract from within it
            u_part_lower = u_part.lower()
            has_non_model_prefix = False
            for prefix in ["bias-eval-results", "worldintheloop", "mutualpredictstrategy"]:
                if prefix in u_part_lower:
                    has_non_model_prefix = True
                    break

            if not has_non_model_prefix and is_likely_policy_name(u_part):
                model_names.append(u_part)
            else:
                # Try to extract model names from hyphenated strings
                # Split by hyphen and look for model patterns
                components = u_part.split("-")
                j = 0
                while j < len(components):
                    comp = components[j]
                    comp_lower = comp.lower()

                    # Check if this component starts a model name
                    model_patterns = [
                        "gpt",
                        "claude",
                        "gemini",
                        "llama",
                        "deepseek",
                        "qwen",
                        "kimi",
                        "mistral",
                        "mixtral",
                        "phi",
                        "falcon",
                        "vicuna",
                        "alpaca",
                        "wizardlm",
                        "yi",
                        "baichuan",
                        "chatglm",
                        "glm",
                        "internlm",
                        "aquila",
                        "bloom",
                        "opt",
                        "galactica",
                    ]

                    is_model_start = False
                    for pattern in model_patterns:
                        if comp_lower.startswith(pattern):
                            is_model_start = True
                            break

                    if is_model_start:
                        # Found a model pattern, collect all parts that belong to this model
                        model_parts = [comp]
                        k = j + 1

                        while k < len(components):
                            next_comp = components[k]
                            next_lower = next_comp.lower()

                            # Stop if we hit a known non-model suffix
                            if next_lower in non_model_suffixes:
                                break

                            # Include if it's a known continuation, version number, or size
                            if (
                                next_lower in model_continuations
                                or re.match(r"^[0-9]", next_comp)  # Starts with number (version)
                                or re.match(r"^[0-9]+[bBmMkK]", next_comp)  # Size like 3B, 70M
                                or re.match(r"^v[0-9]", next_lower)  # Version like v3
                                or re.match(r"^o[0-9]", next_lower)  # OpenAI o1, o3 style
                                or re.match(r"^[0-9]+\.[0-9]+", next_comp)  # Version like 3.1
                                or
                                # Year/date patterns that are often part of model names
                                re.match(r"^20[0-9]{2}", next_comp)  # Year like 2024, 2503
                                or
                                # Single letters that might be part of model names (like V3, A3B)
                                (len(next_comp) == 1 and next_comp.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                                or
                                # Common in model names
                                next_lower in ["base", "instruct", "chat"]
                            ):
                                model_parts.append(next_comp)
                                k += 1
                            else:
                                # Stop if it doesn't look like part of the model name
                                break

                        # Reconstruct the model name
                        full_model_name = "-".join(model_parts)
                        if is_likely_policy_name(full_model_name):
                            model_names.append(full_model_name)
                            j = k  # Skip past the parts we've already processed
                            continue

                    j += 1

    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in model_names:
        # Check if this is a truncated version of an existing name
        # or if an existing name is a truncated version of this
        is_duplicate = False
        for existing in list(seen):
            if name in existing or existing in name:
                # Keep the longer version
                if len(name) > len(existing):
                    seen.remove(existing)
                    unique_names = [n for n in unique_names if n != existing]
                else:
                    is_duplicate = True
                    break

        if not is_duplicate and name not in seen:
            seen.add(name)
            unique_names.append(name)

    return unique_names
