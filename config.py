"""
config for trifetch
can override with TRIFETCH_ env vars
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


class ModelBackend(Enum):
    """model backends"""
    LOCAL_TRANSFORMERS = "local_transformers"
    VLLM = "vllm"
    OPENAI = "openai"
    GROQ = "groq"


class LogProbMode(Enum):
    """how to aggregate logprobs"""
    SUM = "sum"
    LENGTH_NORMALIZED = "length_normalized"  # mean


@dataclass
class ModelConfig:
    """model settings"""
    backend: ModelBackend = ModelBackend.LOCAL_TRANSFORMERS

    # local stuff
    local_model_name: str = "distilgpt2"  # small and fast
    local_device: str = "auto"

    # api stuff
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_model_name: Optional[str] = None

    # generation params
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

    def __post_init__(self):
        # env overrides
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
            # auto setup for groq
            self.backend = ModelBackend.GROQ
            self.api_key = os.environ["GROQ_API_KEY"]
            self.api_model_name = self.api_model_name or "llama-3.1-8b-instant"


@dataclass
class SamplerConfig:
    """rejection sampling settings"""
    max_attempts_per_trace: int = 50
    total_max_attempts: int = 200
    required_traces: int = 3

    # distinctness
    min_jaccard_distance: float = 0.3
    use_embedding_similarity: bool = False
    max_embedding_similarity: float = 0.85

    cache_dir: str = ".trace_cache"
    post_generation_sleep_seconds: float = 0.0  # for rate limiting


@dataclass
class DPOConfig:
    """dpo settings"""
    beta: float = 0.1  # kl penalty

    log_prob_mode: LogProbMode = LogProbMode.LENGTH_NORMALIZED

    # length scaling (optional)
    use_length_scaling: bool = False
    length_scaling_factor: float = 0.01

    # reference model
    use_pretrained_reference: bool = True
    reference_model_seed: int = 42

    eps: float = 1e-8


@dataclass
class GRPOConfig:
    """grpo settings"""
    # rank rewards
    reward_best: float = 1.0
    reward_middle: float = 0.5
    reward_worst: float = 0.0

    log_prob_mode: LogProbMode = LogProbMode.LENGTH_NORMALIZED
    eps: float = 1e-8

    # exponential decay (optional)
    use_exponential_decay: bool = False
    decay_rate: float = 0.5

    # saftey shaping for medical stuff
    use_safety_shaping: bool = False
    safety_keywords: list = field(default_factory=lambda: [
        "consult", "emergency", "immediate", "urgent", "refer",
        "hospitalize", "monitor closely", "seek medical attention"
    ])
    safety_bonus: float = 0.2


@dataclass
class Config:
    """main config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    data_dir: str = "."
    sample_files: list = field(default_factory=lambda: [
        "sample1.json", "sample2.json", "sample3.json",
        "sample4.json", "sample5.json"
    ])

    logprob_cache_file: str = ".logprob_cache.json"


def get_config() -> Config:
    """get config"""
    return Config()
