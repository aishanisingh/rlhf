"""
trace generation with rejection sampling

generates multiple distinct traces that all have the correct answer
keeps trying until we get good ones
"""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any

from config import Config, SamplerConfig
from model_interface import ModelInterface, GenerationResult


@dataclass
class MedicalQASample:
    """medical qa sample"""
    sample_id: str
    question: str
    answer: str
    context: str = ""
    options: Dict[str, str] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """a trace with metadata"""
    trace_id: str
    text: str
    extracted_answer: str
    is_correct: bool
    normalized_hash: str
    token_count: int


@dataclass
class VerifiedTraceSet:
    """set of verified traces"""
    sample_id: str
    question: str
    ground_truth: str
    traces: List[ReasoningTrace]
    generation_attempts: int
    cached: bool = False


def load_sample(file_path: str) -> MedicalQASample:
    """load sample from json"""
    with open(file_path, "r") as f:
        data = json.load(f)

    # try different key names
    question_keys = ["Questions", "questions", "Question", "question", "prompt", "text"]
    answer_keys = ["Answer", "answer", "correct_answer", "ground_truth"]

    question = None
    for key in question_keys:
        if key in data:
            question = data[key]
            break

    if question is None:
        raise ValueError(f"cant find question in {file_path}")

    answer = None
    for key in answer_keys:
        if key in data:
            answer = data[key]
            break

    if answer is None:
        raise ValueError(f"cant find answer in {file_path}")

    sample_id = Path(file_path).stem
    options = _parse_options(question)

    return MedicalQASample(
        sample_id=sample_id,
        question=question,
        answer=str(answer).strip().upper(),
        options=options
    )


def _parse_options(question: str) -> Dict[str, str]:
    """extract mcq options"""
    options = {}
    pattern = r"([A-E])[).:]\s*([^\n]+?)(?=\n[A-E][).:]|\n\n|$)"
    matches = re.findall(pattern, question, re.MULTILINE)
    for letter, text in matches:
        options[letter] = text.strip()
    return options


def load_all_samples(config: Config) -> List[MedicalQASample]:
    """load all samples"""
    samples = []
    for filename in config.sample_files:
        file_path = os.path.join(config.data_dir, filename)
        if not os.path.exists(file_path):
            print(f"[sampler] warning: {file_path} not found")
            continue
        try:
            sample = load_sample(file_path)
            samples.append(sample)
            print(f"[sampler] loaded {sample.sample_id}: answer={sample.answer}")
        except Exception as e:
            print(f"[sampler] error loading {file_path}: {e}")
            raise
    return samples


def extract_answer(trace: str, valid_options: Set[str] = None) -> Optional[str]:
    """extract answer from trace"""
    if valid_options is None:
        valid_options = {"A", "B", "C", "D", "E"}

    trace_clean = trace.strip()

    # patterns ordered by specificity
    patterns = [
        r"(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-E])\)?",
        r"correct\s+(?:answer|option)\s*(?:is|:)?\s*\(?([A-E])\)?",
        r"(?:therefore|so|thus|hence)[,:]?\s*\(?([A-E])\)?",
        r"(?:i\s+(?:choose|select|pick|go\s+with)|would\s+be)\s*\(?([A-E])\)?",
        r"(?:option\s+)?([A-E])\s+is\s+(?:the\s+)?correct",
        r"(?:most\s+likely|best\s+answer\s+is)\s*\(?([A-E])\)?",
        r"option\s*\(?([A-E])\)?",
        r"[^\w]([A-E])[).:]",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, trace_clean, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1).upper()
            if answer in valid_options:
                return answer

    # last resort
    last_part = trace_clean[-100:] if len(trace_clean) > 100 else trace_clean
    for letter in valid_options:
        if re.search(rf'\b{letter}\b', last_part):
            return letter

    return None


def compute_normalized_hash(text: str) -> str:
    """hash for distinctness"""
    normalized = text.lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = normalized.strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def tokenize_simple(text: str) -> Set[str]:
    """simple tokenization"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = set(text.split())
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'to', 'of', 'and', 'in', 'that', 'it', 'for', 'on', 'with'}
    return tokens - stopwords


def jaccard_distance(text1: str, text2: str) -> float:
    """jaccard distance (1 = totally different)"""
    tokens1 = tokenize_simple(text1)
    tokens2 = tokenize_simple(text2)

    if not tokens1 and not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    if union == 0:
        return 0.0

    similarity = intersection / union
    return 1.0 - similarity


def is_sufficiently_distinct(
    new_trace: str,
    existing_traces: List[str],
    config: SamplerConfig
) -> Tuple[bool, str]:
    """check if trace is different enough"""
    new_hash = compute_normalized_hash(new_trace)

    for i, existing in enumerate(existing_traces):
        existing_hash = compute_normalized_hash(existing)
        if new_hash == existing_hash:
            return False, f"hash collision with trace {i}"

        distance = jaccard_distance(new_trace, existing)
        if distance < config.min_jaccard_distance:
            return False, f"too similar to trace {i}"

    return True, "ok"


class TraceCache:
    """disk cache for traces"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, sample_id: str) -> Path:
        return self.cache_dir / f"{sample_id}_traces.json"

    def load(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """load cached traces"""
        cache_path = self._get_cache_path(sample_id)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[cache] error loading {sample_id}: {e}")
        return None

    def save(self, sample_id: str, verified_traces: List[Dict], candidates: List[Dict], attempts: int):
        """save traces"""
        cache_path = self._get_cache_path(sample_id)
        data = {
            "sample_id": sample_id,
            "verified_traces": verified_traces,
            "candidates": candidates,
            "attempts": attempts,
            "timestamp": time.time()
        }
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[cache] error saving {sample_id}: {e}")

    def has_complete_traces(self, sample_id: str, required: int) -> bool:
        """check if we have enought traces"""
        cached = self.load(sample_id)
        if cached and len(cached.get("verified_traces", [])) >= required:
            return True
        return False


def create_generation_prompt(sample: MedicalQASample) -> str:
    """prompt for generating trace"""
    prompt = f"""You are a medical expert answering a multiple choice question.

Question:
{sample.question}

Instructions:
1. Think through the question step by step
2. Consider the key clinical findings
3. Evaluate each option
4. End your response with EXACTLY this format: "The answer is X" where X is A, B, C, D, or E

Your reasoning:"""
    return prompt


def generate_traces(
    sample: MedicalQASample,
    model: ModelInterface,
    config: SamplerConfig,
    cache: Optional[TraceCache] = None
) -> VerifiedTraceSet:
    """
    generate traces using rejection sampling

    keeps generating until we get enough traces that:
    1. have correct answer
    2. are different from eachother
    """
    sample_id = sample.sample_id
    print(f"\n[sampler] generating for {sample_id} (answer: {sample.answer})")

    # check cache
    if cache:
        cached_data = cache.load(sample_id)
        if cached_data and len(cached_data.get("verified_traces", [])) >= config.required_traces:
            print(f"[sampler] using {len(cached_data['verified_traces'])} cached traces")
            traces = [
                ReasoningTrace(
                    trace_id=t["trace_id"], text=t["text"],
                    extracted_answer=t["extracted_answer"], is_correct=t["is_correct"],
                    normalized_hash=t["normalized_hash"], token_count=t["token_count"]
                )
                for t in cached_data["verified_traces"][:config.required_traces]
            ]
            return VerifiedTraceSet(
                sample_id=sample_id, question=sample.question,
                ground_truth=sample.answer, traces=traces,
                generation_attempts=cached_data.get("attempts", 0), cached=True
            )

    # init
    verified_traces: List[ReasoningTrace] = []
    verified_texts: List[str] = []
    all_candidates: List[Dict] = []

    # resume from partial cache
    if cache:
        cached_data = cache.load(sample_id)
        if cached_data:
            for t in cached_data.get("verified_traces", []):
                trace = ReasoningTrace(
                    trace_id=t["trace_id"], text=t["text"],
                    extracted_answer=t["extracted_answer"], is_correct=t["is_correct"],
                    normalized_hash=t["normalized_hash"], token_count=t["token_count"]
                )
                verified_traces.append(trace)
                verified_texts.append(t["text"])
            all_candidates = cached_data.get("candidates", [])
            print(f"[sampler] resuming: {len(verified_traces)} verified")

    prompt = create_generation_prompt(sample)
    valid_options = set(sample.options.keys()) if sample.options else {"A", "B", "C", "D", "E"}

    attempts = 0

    while len(verified_traces) < config.required_traces:
        if attempts >= config.total_max_attempts:
            print(f"[sampler] hit max attempts")
            break

        attempts += 1
        print(f"[sampler] attempt {attempts}, have {len(verified_traces)}/{config.required_traces}")

        try:
            result: GenerationResult = model.generate(prompt)
            generated_text = result.text.strip()
        except Exception as e:
            print(f"[sampler] gen error: {e}")
            generated_text = ""

        if config.post_generation_sleep_seconds > 0:
            time.sleep(config.post_generation_sleep_seconds)

        if not generated_text:
            print(f"[sampler] empty generation")
            continue

        extracted_answer = extract_answer(generated_text, valid_options)

        candidate = {
            "text": generated_text,
            "extracted_answer": extracted_answer,
            "attempt": attempts
        }
        all_candidates.append(candidate)

        # check answer
        is_correct = extracted_answer == sample.answer

        if not is_correct:
            if extracted_answer is None:
                print(f"[sampler] couldnt extract answer")
            else:
                print(f"[sampler] wrong: {extracted_answer} != {sample.answer}")
            continue

        # check distinctness
        is_distinct, reason = is_sufficiently_distinct(generated_text, verified_texts, config)

        if not is_distinct:
            print(f"[sampler] not distinct: {reason}")
            continue

        # accept
        trace_id = f"{sample_id}_trace_{len(verified_traces)}"
        trace = ReasoningTrace(
            trace_id=trace_id, text=generated_text,
            extracted_answer=extracted_answer, is_correct=True,
            normalized_hash=compute_normalized_hash(generated_text),
            token_count=len(generated_text.split())
        )

        verified_traces.append(trace)
        verified_texts.append(generated_text)
        print(f"[sampler] accepted {len(verified_traces)}/{config.required_traces}")

        if cache:
            cache.save(
                sample_id,
                [_trace_to_dict(t) for t in verified_traces],
                all_candidates, attempts
            )

    # final save
    if cache:
        cache.save(
            sample_id,
            [_trace_to_dict(t) for t in verified_traces],
            all_candidates, attempts
        )

    return VerifiedTraceSet(
        sample_id=sample_id, question=sample.question,
        ground_truth=sample.answer, traces=verified_traces,
        generation_attempts=attempts, cached=False
    )


def _trace_to_dict(trace: ReasoningTrace) -> Dict:
    """convert to dict"""
    return {
        "trace_id": trace.trace_id,
        "text": trace.text,
        "extracted_answer": trace.extracted_answer,
        "is_correct": trace.is_correct,
        "normalized_hash": trace.normalized_hash,
        "token_count": trace.token_count
    }


def generate_all_traces(
    samples: List[MedicalQASample],
    model: ModelInterface,
    config: Config
) -> Dict[str, VerifiedTraceSet]:
    """generate for all samples"""
    cache = TraceCache(config.sampler.cache_dir)
    results = {}

    for sample in samples:
        trace_set = generate_traces(sample, model, config.sampler, cache)

        if len(trace_set.traces) >= config.sampler.required_traces:
            results[sample.sample_id] = trace_set
            print(f"[sampler] {sample.sample_id}: done ({len(trace_set.traces)} traces)")
        else:
            print(f"[sampler] {sample.sample_id}: incomplete ({len(trace_set.traces)} traces)")
            results[sample.sample_id] = trace_set

    return results


if __name__ == "__main__":
    from config import get_config
    from model_interface import create_model

    config = get_config()

    print("loading samples...")
    samples = load_all_samples(config)
    print(f"loaded {len(samples)} samples")

    print("\ninit model...")
    model = create_model(config.model)

    print("\ngenerating...")
    results = generate_all_traces(samples, model, config)

    print("\n" + "="*60)
    print("summary")
    print("="*60)
    for sample_id, trace_set in results.items():
        status = "done" if len(trace_set.traces) >= config.sampler.required_traces else "incomplete"
        print(f"{sample_id}: {status} ({len(trace_set.traces)} traces)")
