"""
Configuration for TriFetch Online RLHF Workbench.

All settings can be overridden via environment variables with TRIFETCH_ prefix.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


class ModelBackend(Enum):
    """Supported model backends."""
    LOCAL_TRANSFORMERS = "local_transformers"
    VLLM = "vllm"
    OPENAI = "openai"
    GROQ = "groq"


class LogProbMode(Enum):
    """Log probability computation modes."""
    SUM = "sum"  # Sum of token log-probs
    LENGTH_NORMALIZED = "length_normalized"  # Mean of token log-probs


@dataclass
class ModelConfig:
    """Configuration for model backend."""
    backend: ModelBackend = ModelBackend.LOCAL_TRANSFORMERS

    # Local model settings
    local_model_name: str = "Qwen/Qwen2-1.5B-Instruct"  # Instruction-tuned model
    local_device: str = "auto"  # "cuda", "mps", "cpu", or "auto"

    # API settings (for vLLM, OpenAI, Groq)
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_model_name: Optional[str] = None

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

    def __post_init__(self):
        # Override from environment
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
            # Auto-configure for Groq if key is present
            self.backend = ModelBackend.GROQ
            self.api_key = os.environ["GROQ_API_KEY"]
            self.api_model_name = self.api_model_name or "llama-3.1-8b-instant"


@dataclass
class SamplerConfig:
    """Configuration for rejection sampling."""
    max_attempts_per_trace: int = 50  # Max generation attempts per unique trace
    total_max_attempts: int = 200  # Total attempts before giving up
    required_traces: int = 3  # Number of distinct traces needed

    # Distinctness settings
    min_jaccard_distance: float = 0.3  # Minimum token Jaccard distance between traces
    use_embedding_similarity: bool = False  # Use embeddings for distinctness
    max_embedding_similarity: float = 0.85  # Max cosine similarity if using embeddings

    # Cache settings
    cache_dir: str = ".trace_cache"

    # Optional sleep after each model call (for rate limiting, not CUDA)
    post_generation_sleep_seconds: float = 0.0  # Set > 0 for API rate limiting


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""
    beta: float = 0.1  # KL penalty coefficient

    # Log-prob mode
    log_prob_mode: LogProbMode = LogProbMode.LENGTH_NORMALIZED

    # Optional: Token-length-aware beta scaling
    use_length_scaling: bool = False
    length_scaling_factor: float = 0.01  # Additive scaling per token

    # Reference model settings
    use_pretrained_reference: bool = True  # Standard DPO uses pretrained reference
    reference_model_seed: int = 42  # Only used if use_pretrained_reference=False

    # Numerical stability
    eps: float = 1e-8


@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization."""
    # Default rank rewards
    reward_best: float = 1.0
    reward_middle: float = 0.5
    reward_worst: float = 0.0

    # Log-prob mode
    log_prob_mode: LogProbMode = LogProbMode.LENGTH_NORMALIZED

    # Numerical stability
    eps: float = 1e-8

    # Optional: Exponential decay rewards
    use_exponential_decay: bool = False
    decay_rate: float = 0.5  # If enabled: best=1, middle=decay, worst=decay^2

    # Optional: Healthcare safety shaping
    use_safety_shaping: bool = False
    safety_keywords: list = field(default_factory=lambda: [
        "consult", "emergency", "immediate", "urgent", "refer",
        "hospitalize", "monitor closely", "seek medical attention"
    ])
    safety_bonus: float = 0.2  # Bonus for traces containing safety language


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    # Data paths
    data_dir: str = "."
    sample_files: list = field(default_factory=lambda: [
        "sample1.json", "sample2.json", "sample3.json",
        "sample4.json", "sample5.json"
    ])

    # Log-prob cache
    logprob_cache_file: str = ".logprob_cache.json"


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()
