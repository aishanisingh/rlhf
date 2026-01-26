"""
model interface for trifetch
supports local transformers, vllm, openai, groq
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
    """generation output"""
    text: str
    tokens: List[int]
    finish_reason: str


@dataclass
class LogProbResult:
    """logprob output"""
    token_log_probs: List[float]
    tokens: List[int]
    sum_log_prob: float
    mean_log_prob: float
    num_tokens: int


class ModelInterface(ABC):
    """base class for models"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """generate text"""
        pass

    @abstractmethod
    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        mode: LogProbMode = LogProbMode.SUM
    ) -> LogProbResult:
        """get logprob of completion given prompt"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """model metadata"""
        pass


class LocalTransformersModel(ModelInterface):
    """
    local huggingface model
    can use pretrained weights (policy) or random weights (reference)
    """

    def __init__(
        self,
        config: ModelConfig,
        use_random_weights: bool = False,
        random_seed: Optional[int] = None
    ):
        """init model"""
        self.config = config
        self.use_random_weights = use_random_weights

        # figure out device
        if config.local_device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.local_device

        print(f"[model] loading on {self.device}, random_weights={use_random_weights}")

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.local_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # model
        if use_random_weights:
            print(f"[model] using random weights (seed={random_seed})")
            model_config = AutoConfig.from_pretrained(config.local_model_name)
            if random_seed is not None:
                torch.manual_seed(random_seed)
            self.model = AutoModelForCausalLM.from_config(model_config)
            self._init_random_weights(self.model, random_seed)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(config.local_model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _init_random_weights(self, model: nn.Module, seed: Optional[int] = None):
        """random init"""
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
        """generate completion"""
        # use chat template if available
        if self.tokenizer.chat_template is not None and not self.use_random_weights:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][prompt_length:].tolist()
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        finish_reason = "length" if len(new_tokens) >= gen_kwargs["max_new_tokens"] else "stop"

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
        compute logprob of completion given prompt
        does token by token over the completion
        """
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)

        if len(completion_tokens) == 0:
            return LogProbResult(
                token_log_probs=[], tokens=[],
                sum_log_prob=0.0, mean_log_prob=0.0, num_tokens=0
            )

        full_tokens = prompt_tokens + completion_tokens
        input_ids = torch.tensor([full_tokens]).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)

        token_log_probs = []
        prompt_len = len(prompt_tokens)

        for i, token_id in enumerate(completion_tokens):
            pos = prompt_len + i
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
        """model info"""
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
    api backend for openai, vllm, groq etc
    """

    def __init__(self, config: ModelConfig):
        """init client"""
        self.config = config

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("need openai package: pip install openai")

        if config.backend == ModelBackend.GROQ:
            api_key = config.api_key or os.environ.get("GROQ_API_KEY")
            base_url = config.api_base_url or "https://api.groq.com/openai/v1"
        else:
            api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
            base_url = config.api_base_url

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = config.api_model_name or "gpt-3.5-turbo"

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """generate via api"""
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
                tokens=[],
                finish_reason=choice.finish_reason or "stop"
            )
        except Exception as e:
            print(f"[api] error: {e}")
            return GenerationResult(text="", tokens=[], finish_reason=f"error: {str(e)}")

    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        mode: LogProbMode = LogProbMode.SUM
    ) -> LogProbResult:
        """
        get logprobs from api
        not all apis support this so we estimate if needed
        """
        try:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages[:-1],
                max_tokens=len(completion.split()) * 2,
                temperature=0.0,
                logprobs=True,
                top_logprobs=1,
            )

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
                        token_log_probs=token_log_probs, tokens=[],
                        sum_log_prob=sum_log_prob, mean_log_prob=mean_log_prob,
                        num_tokens=len(token_log_probs)
                    )

            # fallback estimate
            num_tokens = len(completion.split())
            return LogProbResult(
                token_log_probs=[], tokens=[],
                sum_log_prob=-2.0 * num_tokens, mean_log_prob=-2.0,
                num_tokens=num_tokens
            )

        except Exception as e:
            print(f"[api] logprob error: {e}")
            num_tokens = len(completion.split())
            return LogProbResult(
                token_log_probs=[], tokens=[],
                sum_log_prob=-2.0 * num_tokens, mean_log_prob=-2.0,
                num_tokens=num_tokens
            )

    def get_model_info(self) -> Dict[str, Any]:
        """model info"""
        return {
            "backend": "openai_compatible",
            "model_name": self.model_name,
            "api_base": self.config.api_base_url,
        }


class LogProbCache:
    """
    cache for logprobs to avoid recomputing
    uses hash of prompt+completion
    """

    def __init__(self, cache_file: str = ".logprob_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        """load from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
                print(f"[cache] loaded {len(self.cache)} entries")
            except Exception as e:
                print(f"[cache] load error: {e}")
                self.cache = {}

    def _save_cache(self):
        """save to disk"""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"[cache] save error: {e}")

    def _compute_key(self, prompt: str, completion: str, model_id: str) -> str:
        """hash key"""
        content = f"{model_id}||{prompt}||{completion}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, prompt: str, completion: str, model_id: str) -> Optional[LogProbResult]:
        """get from cache"""
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

    def put(self, prompt: str, completion: str, model_id: str, result: LogProbResult):
        """store in cache"""
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
    """wrapper that adds caching"""

    def __init__(self, model: ModelInterface, cache: LogProbCache):
        self.model = model
        self.cache = cache
        self.model_id = self._compute_model_id()

    def _compute_model_id(self) -> str:
        """unique id for this model"""
        info = self.model.get_model_info()
        content = json.dumps(info, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """generate (not cached)"""
        return self.model.generate(prompt, **kwargs)

    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        mode: LogProbMode = LogProbMode.SUM
    ) -> LogProbResult:
        """compute with caching"""
        cached = self.cache.get(prompt, completion, self.model_id)
        if cached is not None:
            return cached

        result = self.model.compute_log_prob(prompt, completion, mode)
        self.cache.put(prompt, completion, self.model_id, result)
        return result

    def get_model_info(self) -> Dict[str, Any]:
        """model info"""
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
    factory to create model
    use_random_weights=True for reference model in dpo
    """
    if config.backend == ModelBackend.LOCAL_TRANSFORMERS:
        model = LocalTransformersModel(
            config, use_random_weights=use_random_weights, random_seed=random_seed
        )
    elif config.backend in (ModelBackend.OPENAI, ModelBackend.VLLM, ModelBackend.GROQ):
        if use_random_weights:
            raise ValueError("cant use random weights with api backend")
        model = OpenAICompatibleModel(config)
    else:
        raise ValueError(f"unknown backend: {config.backend}")

    if enable_cache:
        cache = LogProbCache(cache_file)
        model = CachedModelInterface(model, cache)

    return model
