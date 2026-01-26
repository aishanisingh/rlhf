"""
dpo and grpo optimization for trifetch

dpo = direct preference optimization (skips reward model, uses log-prob ratios)
grpo = group relative policy optimization (compares all 3 traces not just pairs)

basically: pretrain -> sft -> dpo/grpo
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum

from config import Config, DPOConfig, GRPOConfig, LogProbMode
from model_interface import ModelInterface


class Rank(Enum):
    """ranking options"""
    BEST = "best"
    MIDDLE = "middle"
    WORST = "worst"


@dataclass
class RankedTrace:
    """trace with its rank"""
    trace_id: str
    text: str
    rank: Rank


@dataclass
class LogProbPair:
    """log probs from both models"""
    policy_sum: float
    policy_mean: float
    policy_num_tokens: int
    reference_sum: float
    reference_mean: float
    reference_num_tokens: int

    def get_policy_logprob(self, mode: LogProbMode) -> float:
        if mode == LogProbMode.SUM:
            return self.policy_sum
        return self.policy_mean

    def get_reference_logprob(self, mode: LogProbMode) -> float:
        if mode == LogProbMode.SUM:
            return self.reference_sum
        return self.reference_mean


@dataclass
class DPOResult:
    """dpo output"""
    loss: float
    margin: float  # chosen - rejected

    # implicit rewards
    chosen_reward: float
    rejected_reward: float

    # raw logprobs
    chosen_logprobs: LogProbPair
    rejected_logprobs: LogProbPair

    # config stuff
    beta: float
    effective_beta: float
    log_prob_mode: LogProbMode

    # ids
    chosen_trace_id: str
    rejected_trace_id: str


@dataclass
class GRPOResult:
    """grpo output"""
    advantages: Dict[str, float]
    rewards: Dict[str, float]
    mean_reward: float
    std_reward: float
    safety_bonuses: Dict[str, float] = field(default_factory=dict)
    log_prob_mode: LogProbMode = LogProbMode.SUM
    trace_ids: List[str] = field(default_factory=list)


def stable_log_sigmoid(x: float, eps: float = 1e-8) -> float:
    """
    stable version of log(sigmoid(x))
    avoids overflow for negative x
    """
    if x >= 0:
        return -math.log(1 + math.exp(-x) + eps)
    else:
        return x - math.log(1 + math.exp(x) + eps)


def compute_log_probs_for_trace(
    prompt: str,
    trace_text: str,
    policy_model: ModelInterface,
    reference_model: ModelInterface,
    mode: LogProbMode = LogProbMode.SUM
) -> LogProbPair:
    """get logprobs for trace under both models"""
    policy_result = policy_model.compute_log_prob(prompt, trace_text, mode)
    reference_result = reference_model.compute_log_prob(prompt, trace_text, mode)

    return LogProbPair(
        policy_sum=policy_result.sum_log_prob,
        policy_mean=policy_result.mean_log_prob,
        policy_num_tokens=policy_result.num_tokens,
        reference_sum=reference_result.sum_log_prob,
        reference_mean=reference_result.mean_log_prob,
        reference_num_tokens=reference_result.num_tokens
    )


def compute_dpo_loss(
    prompt: str,
    chosen_trace: RankedTrace,
    rejected_trace: RankedTrace,
    policy_model: ModelInterface,
    reference_model: ModelInterface,
    config: DPOConfig
) -> DPOResult:
    """
    computes dpo loss for a preference pair

    the main idea (from rafailov et al 2023):
    - implicit reward r(x,y) = log(pi_policy) - log(pi_ref)
    - loss = -log(sigmoid(beta * margin))
    - margin = r_chosen - r_rejected

    positive margin = model prefers chosen (good)
    negative margin = model prefers rejected (needs training)

    beta controls how agressive the learning is

    we use rejection sampling so both traces have correct answers,
    we're just comparing reasoning quality not correctness
    """
    # get logprobs for both
    chosen_logprobs = compute_log_probs_for_trace(
        prompt, chosen_trace.text, policy_model, reference_model, config.log_prob_mode
    )
    rejected_logprobs = compute_log_probs_for_trace(
        prompt, rejected_trace.text, policy_model, reference_model, config.log_prob_mode
    )

    mode = config.log_prob_mode

    # implicit rewards: r = log(pi) - log(pi_ref)
    chosen_policy_lp = chosen_logprobs.get_policy_logprob(mode)
    chosen_ref_lp = chosen_logprobs.get_reference_logprob(mode)
    chosen_reward = chosen_policy_lp - chosen_ref_lp

    rejected_policy_lp = rejected_logprobs.get_policy_logprob(mode)
    rejected_ref_lp = rejected_logprobs.get_reference_logprob(mode)
    rejected_reward = rejected_policy_lp - rejected_ref_lp

    # length scaling (optional)
    effective_beta = config.beta
    if config.use_length_scaling:
        avg_tokens = (chosen_logprobs.policy_num_tokens + rejected_logprobs.policy_num_tokens) / 2
        effective_beta = config.beta + config.length_scaling_factor * avg_tokens

    # margin and loss
    margin = chosen_reward - rejected_reward
    loss = -stable_log_sigmoid(effective_beta * margin, config.eps)

    return DPOResult(
        loss=loss,
        margin=margin,
        chosen_reward=chosen_reward,
        rejected_reward=rejected_reward,
        chosen_logprobs=chosen_logprobs,
        rejected_logprobs=rejected_logprobs,
        beta=config.beta,
        effective_beta=effective_beta,
        log_prob_mode=mode,
        chosen_trace_id=chosen_trace.trace_id,
        rejected_trace_id=rejected_trace.trace_id
    )


def compute_grpo_advantages(
    traces: List[RankedTrace],
    config: GRPOConfig
) -> GRPOResult:
    """
    computes grpo advantages for group of traces

    unlike dpo which only uses best vs worst,
    grpo uses all 3 traces

    how it works:
    1. assign rewards based on rank (best=1, mid=0.5, worst=0)
    2. normalize: advantage = (reward - mean) / std

    positive advantage = better than avg, reinforce
    negative advantage = worse than avg, discourage

    theres also optional safety shaping for medical stuff
    (bonus for saying things like "consult specialist")
    """
    if len(traces) != 3:
        raise ValueError(f"grpo needs exactly 3 traces, got {len(traces)}")

    # rewards from ranks
    rank_to_reward = {
        Rank.BEST: config.reward_best,
        Rank.MIDDLE: config.reward_middle,
        Rank.WORST: config.reward_worst
    }

    # exponential decay option
    if config.use_exponential_decay:
        rank_to_reward = {
            Rank.BEST: 1.0,
            Rank.MIDDLE: config.decay_rate,
            Rank.WORST: config.decay_rate ** 2
        }

    rewards = {}
    safety_bonuses = {}

    for trace in traces:
        base_reward = rank_to_reward[trace.rank]

        # safety bonus for medical escalation language
        safety_bonus = 0.0
        if config.use_safety_shaping:
            trace_lower = trace.text.lower()
            for keyword in config.safety_keywords:
                if keyword.lower() in trace_lower:
                    safety_bonus = config.safety_bonus
                    break

        rewards[trace.trace_id] = base_reward + safety_bonus
        safety_bonuses[trace.trace_id] = safety_bonus

    # calc mean and std
    reward_values = list(rewards.values())
    mean_reward = sum(reward_values) / len(reward_values)
    variance = sum((r - mean_reward) ** 2 for r in reward_values) / len(reward_values)
    std_reward = math.sqrt(variance)

    # normalized advantages
    advantages = {}
    for trace_id, reward in rewards.items():
        advantages[trace_id] = (reward - mean_reward) / (std_reward + config.eps)

    return GRPOResult(
        advantages=advantages,
        rewards=rewards,
        mean_reward=mean_reward,
        std_reward=std_reward,
        safety_bonuses=safety_bonuses if config.use_safety_shaping else {},
        log_prob_mode=config.log_prob_mode,
        trace_ids=[t.trace_id for t in traces]
    )


@dataclass
class OptimizationResult:
    """combined results"""
    dpo: DPOResult
    grpo: GRPOResult
    sample_id: str
    prompt: str


class Optimizer:
    """
    main optimizer class

    computes dpo and grpo signals that would be used
    to update model weights during actual training

    we dont actually train here, just compute the signals
    so we can see whats happening without needing a gpu
    """

    def __init__(
        self,
        policy_model: ModelInterface,
        reference_model: ModelInterface,
        config: Config
    ):
        """setup with models and config"""
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config

    def compute_optimization_signals(
        self,
        sample_id: str,
        prompt: str,
        traces: List[RankedTrace]
    ) -> OptimizationResult:
        """compute dpo and grpo for the ranked traces"""
        if len(traces) != 3:
            raise ValueError(f"expected 3 traces, got {len(traces)}")

        # find each rank
        best_trace = None
        middle_trace = None
        worst_trace = None

        for trace in traces:
            if trace.rank == Rank.BEST:
                best_trace = trace
            elif trace.rank == Rank.MIDDLE:
                middle_trace = trace
            elif trace.rank == Rank.WORST:
                worst_trace = trace

        if not all([best_trace, middle_trace, worst_trace]):
            raise ValueError("need one trace of each rank")

        # dpo uses best vs worst
        dpo_result = compute_dpo_loss(
            prompt=prompt,
            chosen_trace=best_trace,
            rejected_trace=worst_trace,
            policy_model=self.policy_model,
            reference_model=self.reference_model,
            config=self.config.dpo
        )

        # grpo uses all three
        grpo_result = compute_grpo_advantages(
            traces=traces,
            config=self.config.grpo
        )

        return OptimizationResult(
            dpo=dpo_result,
            grpo=grpo_result,
            sample_id=sample_id,
            prompt=prompt
        )

    def format_results(self, result: OptimizationResult) -> str:
        """format for display"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"optimization results - {result.sample_id}")
        lines.append("=" * 60)

        # dpo section
        lines.append("\n--- dpo ---")
        lines.append(f"beta: {result.dpo.beta:.4f} (effective: {result.dpo.effective_beta:.4f})")
        lines.append(f"log-prob mode: {result.dpo.log_prob_mode.value}")
        lines.append("")
        lines.append(f"chosen: {result.dpo.chosen_trace_id}")
        lines.append(f"  policy log-prob: {result.dpo.chosen_logprobs.policy_mean:.4f}")
        lines.append(f"  reference log-prob: {result.dpo.chosen_logprobs.reference_mean:.4f}")
        lines.append(f"  implicit reward: {result.dpo.chosen_reward:.4f}")
        lines.append("")
        lines.append(f"rejected: {result.dpo.rejected_trace_id}")
        lines.append(f"  policy log-prob: {result.dpo.rejected_logprobs.policy_mean:.4f}")
        lines.append(f"  reference log-prob: {result.dpo.rejected_logprobs.reference_mean:.4f}")
        lines.append(f"  implicit reward: {result.dpo.rejected_reward:.4f}")
        lines.append("")
        lines.append(f"margin: {result.dpo.margin:.4f}")
        lines.append(f"loss: {result.dpo.loss:.4f}")

        # grpo section
        lines.append("\n--- grpo ---")
        lines.append(f"mean reward: {result.grpo.mean_reward:.4f}")
        lines.append(f"std reward: {result.grpo.std_reward:.4f}")
        lines.append("")
        for trace_id in result.grpo.trace_ids:
            reward = result.grpo.rewards[trace_id]
            advantage = result.grpo.advantages[trace_id]
            safety = result.grpo.safety_bonuses.get(trace_id, 0)
            safety_str = f" (+{safety:.2f} safety)" if safety > 0 else ""
            lines.append(f"  {trace_id}: reward={reward:.2f}{safety_str}, advantage={advantage:+.4f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# test
if __name__ == "__main__":
    from config import get_config
    from model_interface import create_model

    config = get_config()

    print("loading policy model...")
    policy_model = create_model(config.model, use_random_weights=False)

    print("loading reference model...")
    reference_model = create_model(
        config.model,
        use_random_weights=True,
        random_seed=config.dpo.reference_model_seed
    )

    test_traces = [
        RankedTrace(
            trace_id="trace_0",
            text="let me think step by step. the symptoms suggest... the answer is B.",
            rank=Rank.BEST
        ),
        RankedTrace(
            trace_id="trace_1",
            text="this is about cardiac issues. after analysis, the answer is B.",
            rank=Rank.MIDDLE
        ),
        RankedTrace(
            trace_id="trace_2",
            text="cardiovascular problem. answer is B.",
            rank=Rank.WORST
        ),
    ]

    test_prompt = "what cardiac abnormality explains these findings? A) MVP B) PFO C) HCM D) VSD"

    print("\ncomputing signals...")
    optimizer = Optimizer(policy_model, reference_model, config)
    result = optimizer.compute_optimization_signals(
        sample_id="test",
        prompt=test_prompt,
        traces=test_traces
    )

    print(optimizer.format_results(result))
