"""
dpo and grpo optimization logic for trifetch.

two post-training methods for learning from human preferences:

dpo (direct preference optimization):
  - skips the reward model - derives rewards from log-prob ratios
  - loss = -log(sigmoid(beta * (r_chosen - r_rejected)))
  - simpler than ppo: no value function, no clipping, fewer hyperparameters

grpo (group relative policy optimization):
  - works with groups of responses, not just pairs
  - advantage = (reward - mean) / std
  - uses all responses, not just best vs worst

healthcare context:
  - we rank reasoning quality, not answer correctness
  - all traces have correct answers (rejection sampling)
  - optional safety bonus for clinical escalation language

pipeline: pretrain -> sft -> dpo/grpo
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from config import Config, DPOConfig, GRPOConfig, LogProbMode
from model_interface import ModelInterface, LogProbResult


class Rank(Enum):
    """trace ranking from human feedback."""
    BEST = "best"
    MIDDLE = "middle"
    WORST = "worst"


@dataclass
class RankedTrace:
    """a trace with its human-assigned rank."""
    trace_id: str
    text: str
    rank: Rank


@dataclass
class LogProbPair:
    """log-probs under policy and reference models."""
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
    """dpo computation output."""
    loss: float
    margin: float  # chosen_reward - rejected_reward

    # implicit rewards: log(pi) - log(pi_ref)
    chosen_reward: float
    rejected_reward: float

    # raw log-probs
    chosen_logprobs: LogProbPair
    rejected_logprobs: LogProbPair

    # config
    beta: float
    effective_beta: float
    log_prob_mode: LogProbMode

    # metadata
    chosen_trace_id: str
    rejected_trace_id: str


@dataclass
class GRPOResult:
    """grpo computation output."""
    advantages: Dict[str, float]  # trace_id -> advantage
    rewards: Dict[str, float]  # trace_id -> reward
    mean_reward: float
    std_reward: float
    safety_bonuses: Dict[str, float] = field(default_factory=dict)
    log_prob_mode: LogProbMode = LogProbMode.SUM
    trace_ids: List[str] = field(default_factory=list)


def stable_log_sigmoid(x: float, eps: float = 1e-8) -> float:
    """
    numerically stable log(sigmoid(x)).

    log(sigmoid(x)) = -log(1 + exp(-x))

    but for x < 0, we rewrite as x - log(1 + exp(x)) to avoid overflow.
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
    """get log-probs for a trace under both models."""
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
    compute dpo loss for a preference pair.

    the key insight of dpo (rafailov et al. 2023):
      instead of training a reward model then doing rl,
      we can optimize preferences directly using log-prob ratios.

    implicit reward:
      r(x, y) = log(pi_policy(y|x)) - log(pi_ref(y|x))

      this measures how much the policy prefers y compared to the reference.
      if policy likes it more than reference does, reward is positive.

    loss:
      loss = -log(sigmoid(beta * margin))
      margin = r_chosen - r_rejected

      when margin > 0: policy prefers chosen -> low loss (good)
      when margin < 0: policy prefers rejected -> high loss (needs training)

    beta controls how sharp the preference learning is.
    higher beta = more aggressive preference updates.

    why dpo over ppo:
      - no reward model to train
      - no value function to estimate
      - no clipping heuristics
      - direct optimization of bradley-terry preference model
      - more stable, fewer hyperparameters

    healthcare note:
      both traces have correct answers (we used rejection sampling).
      we're comparing reasoning quality, not correctness.
      a model that gets right answers for wrong reasons will fail on edge cases.
    """
    # get log-probs for both traces
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

    # optional length scaling for beta
    effective_beta = config.beta
    if config.use_length_scaling:
        avg_tokens = (chosen_logprobs.policy_num_tokens + rejected_logprobs.policy_num_tokens) / 2
        effective_beta = config.beta + config.length_scaling_factor * avg_tokens

    # margin: how much more the policy rewards chosen over rejected
    margin = chosen_reward - rejected_reward

    # dpo loss: -log(sigmoid(beta * margin))
    # low when margin > 0 (correct preference), high when margin < 0
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
    compute grpo advantages for a group of traces.

    grpo extends preference learning to groups (not just pairs).

    how it works:
      1. assign rewards based on rank: best=1.0, middle=0.5, worst=0.0
      2. normalize: advantage = (reward - mean) / std
      3. use advantages to weight policy gradient updates

    why group-relative:
      - dpo only uses best vs worst, ignoring middle
      - grpo uses all responses, more data-efficient
      - std normalization keeps gradients stable across batches

    the advantage tells you:
      positive -> better than average, reinforce this behavior
      negative -> worse than average, discourage this behavior
      magnitude -> how strongly to update

    safety shaping (optional):
      in healthcare, false negatives are worse than false positives.
      we want models that suggest "consult a specialist" when uncertain.
      safety bonus rewards traces with escalation language like
      "emergency", "refer to", "consult", "urgent".
    """
    if len(traces) != 3:
        raise ValueError(f"grpo requires exactly 3 traces, got {len(traces)}")

    # base rewards from ranks
    rank_to_reward = {
        Rank.BEST: config.reward_best,
        Rank.MIDDLE: config.reward_middle,
        Rank.WORST: config.reward_worst
    }

    # optional exponential decay: best=1, middle=decay, worst=decay^2
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

        # safety shaping: bonus for clinical escalation language
        # in medical ai, we want models that err on the side of caution
        safety_bonus = 0.0
        if config.use_safety_shaping:
            trace_lower = trace.text.lower()
            for keyword in config.safety_keywords:
                if keyword.lower() in trace_lower:
                    safety_bonus = config.safety_bonus
                    break

        rewards[trace.trace_id] = base_reward + safety_bonus
        safety_bonuses[trace.trace_id] = safety_bonus

    # group statistics
    reward_values = list(rewards.values())
    mean_reward = sum(reward_values) / len(reward_values)
    variance = sum((r - mean_reward) ** 2 for r in reward_values) / len(reward_values)
    std_reward = math.sqrt(variance)

    # advantages: normalized rewards
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
    """combined dpo and grpo results."""
    dpo: DPOResult
    grpo: GRPOResult
    sample_id: str
    prompt: str


class Optimizer:
    """
    computes dpo and grpo optimization signals.

    this is the core of post-training preference learning.
    the signals computed here are exactly what would update model weights
    during actual training.

    we compute signals without training to:
      - demonstrate the math clearly
      - allow exploration without gpu
      - focus on preference learning, not training infrastructure

    in a real training loop you would:
      1. compute these signals for a batch
      2. backprop the dpo loss
      3. update weights with adamw or similar
    """

    def __init__(
        self,
        policy_model: ModelInterface,
        reference_model: ModelInterface,
        config: Config
    ):
        """
        set up optimizer with models and config.

        the reference model provides the baseline for kl divergence.
        in standard dpo, this is the sft model (after supervised fine-tuning,
        before preference optimization).
        """
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config

    def compute_optimization_signals(
        self,
        sample_id: str,
        prompt: str,
        traces: List[RankedTrace]
    ) -> OptimizationResult:
        """compute dpo and grpo signals for ranked traces."""
        if len(traces) != 3:
            raise ValueError(f"expected 3 traces, got {len(traces)}")

        # find traces by rank
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
            raise ValueError("need exactly one trace of each rank")

        # dpo: best vs worst
        dpo_result = compute_dpo_loss(
            prompt=prompt,
            chosen_trace=best_trace,
            rejected_trace=worst_trace,
            policy_model=self.policy_model,
            reference_model=self.reference_model,
            config=self.config.dpo
        )

        # grpo: all three traces
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
        """format results for display."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"optimization results - {result.sample_id}")
        lines.append("=" * 60)

        # dpo
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

        # grpo
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


# command-line test
if __name__ == "__main__":
    from config import get_config
    from model_interface import create_model

    config = get_config()

    print("loading policy model...")
    policy_model = create_model(config.model, use_random_weights=False)

    print("loading reference model (random weights)...")
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

    print("\ncomputing optimization signals...")
    optimizer = Optimizer(policy_model, reference_model, config)
    result = optimizer.compute_optimization_signals(
        sample_id="test",
        prompt=test_prompt,
        traces=test_traces
    )

    print(optimizer.format_results(result))
