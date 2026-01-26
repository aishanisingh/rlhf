"""
config for trifetch.

all settings can be overridden via environment variables with TRIFETCH_ prefix.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


class ModelBackend(Enum):
    """supported model backends."""
    LOCAL_TRANSFORMERS = "local_transformers"
    VLLM = "vllm"
    OPENAI = "openai"
    GROQ = "groq"


class LogProbMode(Enum):
    """how to aggregate token log-probs."""
    SUM = "sum"  # sum of token log-probs
    LENGTH_NORMALIZED = "length_normalized"  # mean of token log-probs


@dataclass
class ModelConfig:
    """model backend settings."""
    backend: ModelBackend = ModelBackend.LOCAL_TRANSFORMERS

    # local model settings
    local_model_name: str = "distilgpt2"  # small and fast (~82MB)
    local_device: str = "auto"  # "cuda", "mps", "cpu", or "auto"

    # api settings (for vllm, openai, groq)
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_model_name: Optional[str] = None

    # generation settings
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

    def __post_init__(self):
        # override from environment
        if os.environ.get("TRIFETCH_MODEL_BACKEND"):
            self.backend = ModelBackend(os.environ["TRIFETCH_MODEL_BACKEND"])
        if os.environ.get("TRIFETCH_LOCAL_MODEL"):
            self.local_model_name = os.environ["TRIFETCH_LOCAL_MODEL"]
        if os.environ.get("TRIFETCH_API_BASE_URL"):
            self.api_base_url = os.environ["TRIFETCH_API_BASE_URL"]
        if os.environ.get("TRIFETCH_API_KEY"):
            self.api_key = os.environ["TRIFETCH_API_KEY"]
        if os.environ.get("TRIFETCH_API_MODEL"):
            self.api_model_name = os.environ["TRIFETCH_API_MODEL"]
        if os.environ.get("GROQ_API_KEY"):
            # auto-configure for groq if key is present
            self.backend = ModelBackend.GROQ
            self.api_key = os.environ["GROQ_API_KEY"]
            self.api_model_name = self.api_model_name or "llama-3.1-8b-instant"


@dataclass
class SamplerConfig:
    """rejection sampling settings."""
    max_attempts_per_trace: int = 50  # max generation attempts per unique trace
    total_max_attempts: int = 200  # total attempts before giving up
    required_traces: int = 3  # number of distinct traces needed

    # distinctness settings
    min_jaccard_distance: float = 0.3  # minimum token jaccard distance between traces
    use_embedding_similarity: bool = False  # use embeddings for distinctness
    max_embedding_similarity: float = 0.85  # max cosine similarity if using embeddings

    # cache settings
    cache_dir: str = ".trace_cache"

    # optional sleep after each model call (for rate limiting)
    post_generation_sleep_seconds: float = 0.0  # set > 0 for api rate limiting


@dataclass
class DPOConfig:
    """dpo settings - direct preference optimization."""
    beta: float = 0.1  # kl penalty coefficient

    # log-prob mode
    log_prob_mode: LogProbMode = LogProbMode.LENGTH_NORMALIZED

    # optional: token-length-aware beta scaling
    use_length_scaling: bool = False
    length_scaling_factor: float = 0.01  # additive scaling per token

    # reference model settings
    use_pretrained_reference: bool = True  # standard dpo uses pretrained reference
    reference_model_seed: int = 42  # only used if use_pretrained_reference=False

    # numerical stability
    eps: float = 1e-8


@dataclass
class GRPOConfig:
    """grpo settings - group relative policy optimization."""
    # default rank rewards
    reward_best: float = 1.0
    reward_middle: float = 0.5
    reward_worst: float = 0.0

    # log-prob mode
    log_prob_mode: LogProbMode = LogProbMode.LENGTH_NORMALIZED

    # numerical stability
    eps: float = 1e-8

    # optional: exponential decay rewards
    use_exponential_decay: bool = False
    decay_rate: float = 0.5  # if enabled: best=1, middle=decay, worst=decay^2

    # optional: healthcare safety shaping
    use_safety_shaping: bool = False
    safety_keywords: list = field(default_factory=lambda: [
        "consult", "emergency", "immediate", "urgent", "refer",
        "hospitalize", "monitor closely", "seek medical attention"
    ])
    safety_bonus: float = 0.2  # bonus for traces containing safety language


@dataclass
class Config:
    """main config container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    # data paths
    data_dir: str = "."
    sample_files: list = field(default_factory=lambda: [
        "sample1.json", "sample2.json", "sample3.json",
        "sample4.json", "sample5.json"
    ])

    # log-prob cache
    logprob_cache_file: str = ".logprob_cache.json"


def get_config() -> Config:
    """get the global config instance."""
    return Config()
