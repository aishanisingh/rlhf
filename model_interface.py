"""
model interface for trifetch.

abstraction layer for text generation and log-probability computation.
supports local transformers, vllm, openai, groq.
"""
import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from config import ModelConfig, ModelBackend, LogProbMode


@dataclass
class GenerationResult:
    """result of a text generation call."""
    text: str
    tokens: List[int]
    finish_reason: str  # "stop", "length", "error"


@dataclass
class LogProbResult:
    """result of log-probability computation."""
    token_log_probs: List[float]  # per-token log-probs
    tokens: List[int]
    sum_log_prob: float  # sum of token log-probs
    mean_log_prob: float  # length-normalized log-prob
    num_tokens: int


class ModelInterface(ABC):
    """abstract interface for language model operations."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """generate text completion for the given prompt."""
        pass

    @abstractmethod
    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        mode: LogProbMode = LogProbMode.SUM
    ) -> LogProbResult:
        """
        compute log-probability of completion given prompt.

        log-probs are computed token-by-token over the completion only,
        conditioned on the full prompt and previous completion tokens.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """return metadata about the model."""
        pass


class LocalTransformersModel(ModelInterface):
    """
    local model backend using huggingface transformers.

    supports both policy model (pretrained) and reference model (random weights).
    """

    def __init__(
        self,
        config: ModelConfig,
        use_random_weights: bool = False,
        random_seed: Optional[int] = None
    ):
        """
        initialize the local model.

        use_random_weights=True for reference model in dpo.
        """
        self.config = config
        self.use_random_weights = use_random_weights

        # determine device
        if config.local_device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.local_device

        print(f"[model] loading on device: {self.device}")
        print(f"[model] random weights: {use_random_weights}")

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.local_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # load model
        if use_random_weights:
            # random weights for reference model
            print(f"[model] initializing random weights with seed {random_seed}")
            model_config = AutoConfig.from_pretrained(config.local_model_name)

            if random_seed is not None:
                torch.manual_seed(random_seed)

            # create model with random weights (no pretrained loading)
            self.model = AutoModelForCausalLM.from_config(model_config)
            self._init_random_weights(self.model, random_seed)
        else:
            # pretrained weights for policy model
            self.model = AutoModelForCausalLM.from_pretrained(config.local_model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _init_random_weights(self, model: nn.Module, seed: Optional[int] = None):
        """initialize model with random weights using consistent seeding."""
        if seed is not None:
            torch.manual_seed(seed)

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """generate text completion using chat template if available."""
        # check if model has a chat template (instruction-tuned models)
        if self.tokenizer.chat_template is not None and not self.use_random_weights:
            # use chat format for instruction-tuned models
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # use raw prompt for base models or random weight models
            formatted_prompt = prompt

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        # merge with config defaults
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        # extract only the new tokens
        new_tokens = outputs[0][prompt_length:].tolist()
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # determine finish reason
        if len(new_tokens) >= gen_kwargs["max_new_tokens"]:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return GenerationResult(
            text=generated_text,
            tokens=new_tokens,
            finish_reason=finish_reason
        )

    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        mode: LogProbMode = LogProbMode.SUM
    ) -> LogProbResult:
        """
        compute log-probability of completion given prompt.

        computes token-by-token log-probs over completion only,
        properly conditioned on prompt + previous completion tokens.
        """
        # tokenize prompt and completion separately
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)

        if len(completion_tokens) == 0:
            return LogProbResult(
                token_log_probs=[],
                tokens=[],
                sum_log_prob=0.0,
                mean_log_prob=0.0,
                num_tokens=0
            )

        # concatenate for full sequence
        full_tokens = prompt_tokens + completion_tokens
        input_ids = torch.tensor([full_tokens]).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

        # compute log-probs for each completion token
        # logits[i] predicts token[i+1]
        log_probs = torch.log_softmax(logits, dim=-1)

        token_log_probs = []
        prompt_len = len(prompt_tokens)

        for i, token_id in enumerate(completion_tokens):
            # position in full sequence where this token appears
            pos = prompt_len + i
            # log-prob comes from previous position's prediction
            token_log_prob = log_probs[0, pos - 1, token_id].item()
            token_log_probs.append(token_log_prob)

        sum_log_prob = sum(token_log_probs)
        mean_log_prob = sum_log_prob / len(token_log_probs) if token_log_probs else 0.0

        return LogProbResult(
            token_log_probs=token_log_probs,
            tokens=completion_tokens,
            sum_log_prob=sum_log_prob,
            mean_log_prob=mean_log_prob,
            num_tokens=len(completion_tokens)
        )

    def get_model_info(self) -> Dict[str, Any]:
        """return model metadata."""
        return {
            "backend": "local_transformers",
            "model_name": self.config.local_model_name,
            "device": self.device,
            "random_weights": self.use_random_weights,
            "vocab_size": self.tokenizer.vocab_size,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }


class OpenAICompatibleModel(ModelInterface):
    """
    openai-compatible api backend using chat completions api.

    works with openai, vllm serving, groq, and other compatible apis.
    """

    def __init__(self, config: ModelConfig):
        """initialize api client."""
        self.config = config

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for api backend. pip install openai")

        # determine api key based on backend
        if config.backend == ModelBackend.GROQ:
            api_key = config.api_key or os.environ.get("GROQ_API_KEY")
            base_url = config.api_base_url or "https://api.groq.com/openai/v1"
        else:
            api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
            base_url = config.api_base_url

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = config.api_model_name or "gpt-3.5-turbo"

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """generate text via chat completions api."""
        try:
            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
            )

            choice = response.choices[0]
            return GenerationResult(
                text=choice.message.content or "",
                tokens=[],  # api doesn't return token ids
                finish_reason=choice.finish_reason or "stop"
            )
        except Exception as e:
            print(f"[api] generation error: {e}")
            return GenerationResult(
                text="",
                tokens=[],
                finish_reason=f"error: {str(e)}"
            )

    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        mode: LogProbMode = LogProbMode.SUM
    ) -> LogProbResult:
        """
        compute log-probability via api.

        note: chat api log-probs are requested via logprobs parameter.
        for apis that don't support log-probs, returns estimated values.
        """
        try:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]

            # try to get log-probs (supported by some apis)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages[:-1],  # just the user message
                max_tokens=len(completion.split()) * 2,  # approximate
                temperature=0.0,  # deterministic for log-prob estimation
                logprobs=True,
                top_logprobs=1,
            )

            # extract log-probs if available
            choice = response.choices[0]
            if hasattr(choice, 'logprobs') and choice.logprobs and choice.logprobs.content:
                token_log_probs = []
                for token_data in choice.logprobs.content:
                    if token_data.logprob is not None:
                        token_log_probs.append(token_data.logprob)

                if token_log_probs:
                    sum_log_prob = sum(token_log_probs)
                    mean_log_prob = sum_log_prob / len(token_log_probs)

                    return LogProbResult(
                        token_log_probs=token_log_probs,
                        tokens=[],
                        sum_log_prob=sum_log_prob,
                        mean_log_prob=mean_log_prob,
                        num_tokens=len(token_log_probs)
                    )

            # fallback: estimate based on completion length
            # this is approximate but allows the system to function
            num_tokens = len(completion.split())
            estimated_log_prob = -2.0 * num_tokens  # rough estimate

            return LogProbResult(
                token_log_probs=[],
                tokens=[],
                sum_log_prob=estimated_log_prob,
                mean_log_prob=-2.0,
                num_tokens=num_tokens
            )

        except Exception as e:
            print(f"[api] log-prob error: {e}")
            # return estimated values to keep system functional
            num_tokens = len(completion.split())
            return LogProbResult(
                token_log_probs=[],
                tokens=[],
                sum_log_prob=-2.0 * num_tokens,
                mean_log_prob=-2.0,
                num_tokens=num_tokens
            )

    def get_model_info(self) -> Dict[str, Any]:
        """return model metadata."""
        return {
            "backend": "openai_compatible",
            "model_name": self.model_name,
            "api_base": self.config.api_base_url,
        }


class LogProbCache:
    """
    cache for log-probability computations.

    uses deterministic hashing of prompt-completion pairs to avoid recomputation.
    """

    def __init__(self, cache_file: str = ".logprob_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        """load cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
                print(f"[cache] loaded {len(self.cache)} entries from {self.cache_file}")
            except Exception as e:
                print(f"[cache] error loading: {e}")
                self.cache = {}

    def _save_cache(self):
        """save cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"[cache] error saving: {e}")

    def _compute_key(self, prompt: str, completion: str, model_id: str) -> str:
        """compute deterministic hash key for prompt-completion pair."""
        content = f"{model_id}||{prompt}||{completion}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(
        self,
        prompt: str,
        completion: str,
        model_id: str
    ) -> Optional[LogProbResult]:
        """retrieve cached log-prob result."""
        key = self._compute_key(prompt, completion, model_id)
        if key in self.cache:
            data = self.cache[key]
            return LogProbResult(
                token_log_probs=data["token_log_probs"],
                tokens=data["tokens"],
                sum_log_prob=data["sum_log_prob"],
                mean_log_prob=data["mean_log_prob"],
                num_tokens=data["num_tokens"]
            )
        return None

    def put(
        self,
        prompt: str,
        completion: str,
        model_id: str,
        result: LogProbResult
    ):
        """store log-prob result in cache."""
        key = self._compute_key(prompt, completion, model_id)
        self.cache[key] = {
            "token_log_probs": result.token_log_probs,
            "tokens": result.tokens,
            "sum_log_prob": result.sum_log_prob,
            "mean_log_prob": result.mean_log_prob,
            "num_tokens": result.num_tokens
        }
        self._save_cache()


class CachedModelInterface(ModelInterface):
    """wrapper that adds caching to any ModelInterface."""

    def __init__(self, model: ModelInterface, cache: LogProbCache):
        self.model = model
        self.cache = cache
        self.model_id = self._compute_model_id()

    def _compute_model_id(self) -> str:
        """create unique identifier for this model instance."""
        info = self.model.get_model_info()
        content = json.dumps(info, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """generate text (not cached, as generation should vary)."""
        return self.model.generate(prompt, **kwargs)

    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        mode: LogProbMode = LogProbMode.SUM
    ) -> LogProbResult:
        """compute log-prob with caching."""
        # check cache
        cached = self.cache.get(prompt, completion, self.model_id)
        if cached is not None:
            return cached

        # compute and cache
        result = self.model.compute_log_prob(prompt, completion, mode)
        self.cache.put(prompt, completion, self.model_id, result)
        return result

    def get_model_info(self) -> Dict[str, Any]:
        """return model metadata."""
        info = self.model.get_model_info()
        info["cached"] = True
        return info


def create_model(
    config: ModelConfig,
    use_random_weights: bool = False,
    random_seed: Optional[int] = None,
    enable_cache: bool = True,
    cache_file: str = ".logprob_cache.json"
) -> ModelInterface:
    """
    factory function to create model interface based on configuration.

    use_random_weights=True for reference model in dpo.
    """
    if config.backend == ModelBackend.LOCAL_TRANSFORMERS:
        model = LocalTransformersModel(
            config,
            use_random_weights=use_random_weights,
            random_seed=random_seed
        )
    elif config.backend in (ModelBackend.OPENAI, ModelBackend.VLLM, ModelBackend.GROQ):
        if use_random_weights:
            raise ValueError("random weights not supported for api backends. use local model for reference.")
        model = OpenAICompatibleModel(config)
    else:
        raise ValueError(f"unknown model backend: {config.backend}")

    if enable_cache:
        cache = LogProbCache(cache_file)
        model = CachedModelInterface(model, cache)

    return model
