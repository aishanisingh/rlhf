"""
DPO and GRPO optimization logic for TriFetch RLHF Workbench.

Computes optimization signals based on human preference rankings.
Implements mathematically precise formulations with healthcare-aware extensions.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from config import Config, DPOConfig, GRPOConfig, LogProbMode
from model_interface import ModelInterface, LogProbResult


class Rank(Enum):
    """Trace ranking from human feedback."""
    BEST = "best"
    MIDDLE = "middle"
    WORST = "worst"


@dataclass
class RankedTrace:
    """A trace with its human-assigned rank."""
    trace_id: str
    text: str
    rank: Rank


@dataclass
class LogProbPair:
    """Log-probabilities under policy and reference models."""
    policy_sum: float
    policy_mean: float
    policy_num_tokens: int
    reference_sum: float
    reference_mean: float
    reference_num_tokens: int

    def get_policy_logprob(self, mode: LogProbMode) -> float:
        """Get policy log-prob based on mode."""
        if mode == LogProbMode.SUM:
            return self.policy_sum
        return self.policy_mean

    def get_reference_logprob(self, mode: LogProbMode) -> float:
        """Get reference log-prob based on mode."""
        if mode == LogProbMode.SUM:
            return self.reference_sum
        return self.reference_mean


@dataclass
class DPOResult:
    """Result of DPO computation."""
    # Core DPO outputs
    loss: float
    margin: float  # rejected_reward - chosen_reward (inverted per spec)

    # Implicit rewards
    chosen_reward: float  # policy_logprob - ref_logprob for best trace
    rejected_reward: float  # policy_logprob - ref_logprob for worst trace

    # Log-probabilities
    chosen_logprobs: LogProbPair
    rejected_logprobs: LogProbPair

    # Configuration used
    beta: float
    effective_beta: float  # After any length scaling
    log_prob_mode: LogProbMode

    # Metadata
    chosen_trace_id: str
    rejected_trace_id: str


@dataclass
class GRPOResult:
    """Result of GRPO computation."""
    # Per-trace advantages
    advantages: Dict[str, float]  # trace_id -> advantage

    # Per-trace rewards (before normalization)
    rewards: Dict[str, float]  # trace_id -> reward

    # Group statistics
    mean_reward: float
    std_reward: float

    # Optional: safety-shaped rewards
    safety_bonuses: Dict[str, float] = field(default_factory=dict)

    # Configuration used
    log_prob_mode: LogProbMode = LogProbMode.SUM

    # Metadata
    trace_ids: List[str] = field(default_factory=list)


def stable_log_sigmoid(x: float, eps: float = 1e-8) -> float:
    """
    Numerically stable computation of log(sigmoid(x)).

    log(sigmoid(x)) = log(1 / (1 + exp(-x)))
                    = -log(1 + exp(-x))

    For numerical stability:
    - When x >= 0: -log(1 + exp(-x)) directly
    - When x < 0: x - log(1 + exp(x)) to avoid overflow
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
    """
    Compute log-probabilities for a trace under both policy and reference models.

    Args:
        prompt: The input prompt (question)
        trace_text: The reasoning trace text
        policy_model: The policy model (pretrained weights)
        reference_model: The reference model (random weights)
        mode: Whether to use sum or length-normalized log-probs

    Returns:
        LogProbPair with both policy and reference log-probs
    """
    # Compute policy log-prob
    policy_result = policy_model.compute_log_prob(prompt, trace_text, mode)

    # Compute reference log-prob
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
    Compute Direct Preference Optimization (DPO) loss.

    The DPO loss encourages the policy to prefer chosen over rejected completions
    while staying close to the reference distribution.

    The implicit reward for a completion y given prompt x is:
        r(x, y) = beta * (log pi_policy(y|x) - log pi_ref(y|x))

    The loss is: -log(sigmoid(beta * (r_chosen - r_rejected)))

    This means:
    - When chosen_reward > rejected_reward (good): loss is LOW
    - When rejected_reward > chosen_reward (bad): loss is HIGH

    Args:
        prompt: The input prompt
        chosen_trace: The trace ranked as Best
        rejected_trace: The trace ranked as Worst
        policy_model: Policy model with pretrained weights
        reference_model: Reference model with random weights
        config: DPO configuration

    Returns:
        DPOResult with loss, margin, and all intermediate values
    """
    # Compute log-probs for both traces
    chosen_logprobs = compute_log_probs_for_trace(
        prompt, chosen_trace.text, policy_model, reference_model, config.log_prob_mode
    )
    rejected_logprobs = compute_log_probs_for_trace(
        prompt, rejected_trace.text, policy_model, reference_model, config.log_prob_mode
    )

    # Get log-probs based on configured mode
    mode = config.log_prob_mode

    # Compute implicit rewards: r(x,y) = log(pi(y|x)) - log(pi_ref(y|x))
    chosen_policy_lp = chosen_logprobs.get_policy_logprob(mode)
    chosen_ref_lp = chosen_logprobs.get_reference_logprob(mode)
    chosen_reward = chosen_policy_lp - chosen_ref_lp

    rejected_policy_lp = rejected_logprobs.get_policy_logprob(mode)
    rejected_ref_lp = rejected_logprobs.get_reference_logprob(mode)
    rejected_reward = rejected_policy_lp - rejected_ref_lp

    # Compute effective beta (with optional length scaling)
    effective_beta = config.beta
    if config.use_length_scaling:
        # Scale beta based on average token length of traces
        avg_tokens = (chosen_logprobs.policy_num_tokens + rejected_logprobs.policy_num_tokens) / 2
        effective_beta = config.beta + config.length_scaling_factor * avg_tokens

    # Standard DPO margin: chosen_reward - rejected_reward
    # This rewards positive margins (when chosen is preferred)
    margin = chosen_reward - rejected_reward

    # DPO loss: -log(sigmoid(beta * margin))
    # Loss is LOW when chosen_reward > rejected_reward (correct preference)
    # Loss is HIGH when chosen_reward < rejected_reward (incorrect preference)
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
    Compute Group Relative Policy Optimization (GRPO) advantages.

    GRPO operates over the entire group of traces simultaneously,
    computing group-relative advantages based on rankings.

    Advantage = (reward - mean_reward) / (std_reward + eps)

    Args:
        traces: List of all three ranked traces (Best, Middle, Worst)
        config: GRPO configuration

    Returns:
        GRPOResult with per-trace advantages and group statistics
    """
    if len(traces) != 3:
        raise ValueError(f"GRPO requires exactly 3 traces, got {len(traces)}")

    # Assign rewards based on rank
    rank_to_reward = {
        Rank.BEST: config.reward_best,
        Rank.MIDDLE: config.reward_middle,
        Rank.WORST: config.reward_worst
    }

    # Optional: Use exponential decay rewards
    if config.use_exponential_decay:
        rank_to_reward = {
            Rank.BEST: 1.0,
            Rank.MIDDLE: config.decay_rate,
            Rank.WORST: config.decay_rate ** 2
        }

    # Compute base rewards
    rewards = {}
    safety_bonuses = {}

    for trace in traces:
        base_reward = rank_to_reward[trace.rank]

        # Optional: Healthcare safety shaping
        safety_bonus = 0.0
        if config.use_safety_shaping:
            trace_lower = trace.text.lower()
            for keyword in config.safety_keywords:
                if keyword.lower() in trace_lower:
                    safety_bonus = config.safety_bonus
                    break

        rewards[trace.trace_id] = base_reward + safety_bonus
        safety_bonuses[trace.trace_id] = safety_bonus

    # Compute group statistics
    reward_values = list(rewards.values())
    mean_reward = sum(reward_values) / len(reward_values)

    # Standard deviation
    variance = sum((r - mean_reward) ** 2 for r in reward_values) / len(reward_values)
    std_reward = math.sqrt(variance)

    # Compute advantages: (reward - mean) / (std + eps)
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
    """Combined result of all optimization computations."""
    dpo: DPOResult
    grpo: GRPOResult
    sample_id: str
    prompt: str


class Optimizer:
    """
    Main optimizer class that orchestrates DPO and GRPO computations.

    Encapsulates model references and configuration for easy reuse.
    """

    def __init__(
        self,
        policy_model: ModelInterface,
        reference_model: ModelInterface,
        config: Config
    ):
        """
        Initialize optimizer with models and configuration.

        Args:
            policy_model: Policy model with pretrained weights
            reference_model: Reference model with RANDOM weights (not pretrained)
            config: Full configuration
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
        """
        Compute all optimization signals for a set of ranked traces.

        Args:
            sample_id: Identifier for the sample
            prompt: The input prompt
            traces: List of 3 traces with Best, Middle, Worst rankings

        Returns:
            OptimizationResult containing DPO and GRPO outputs
        """
        # Validate input
        if len(traces) != 3:
            raise ValueError(f"Expected 3 traces, got {len(traces)}")

        # Extract traces by rank
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
            raise ValueError("Must have exactly one trace of each rank (Best, Middle, Worst)")

        # Compute DPO loss (Best vs Worst)
        dpo_result = compute_dpo_loss(
            prompt=prompt,
            chosen_trace=best_trace,
            rejected_trace=worst_trace,
            policy_model=self.policy_model,
            reference_model=self.reference_model,
            config=self.config.dpo
        )

        # Compute GRPO advantages (all three traces)
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
        """Format optimization results for display."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"OPTIMIZATION RESULTS - {result.sample_id}")
        lines.append("=" * 60)

        # DPO Results
        lines.append("\n--- DPO (Direct Preference Optimization) ---")
        lines.append(f"Beta: {result.dpo.beta:.4f} (effective: {result.dpo.effective_beta:.4f})")
        lines.append(f"Log-prob mode: {result.dpo.log_prob_mode.value}")
        lines.append("")
        lines.append(f"Chosen trace: {result.dpo.chosen_trace_id}")
        lines.append(f"  Policy log-prob (sum): {result.dpo.chosen_logprobs.policy_sum:.4f}")
        lines.append(f"  Policy log-prob (mean): {result.dpo.chosen_logprobs.policy_mean:.4f}")
        lines.append(f"  Reference log-prob (sum): {result.dpo.chosen_logprobs.reference_sum:.4f}")
        lines.append(f"  Reference log-prob (mean): {result.dpo.chosen_logprobs.reference_mean:.4f}")
        lines.append(f"  Implicit reward: {result.dpo.chosen_reward:.4f}")
        lines.append("")
        lines.append(f"Rejected trace: {result.dpo.rejected_trace_id}")
        lines.append(f"  Policy log-prob (sum): {result.dpo.rejected_logprobs.policy_sum:.4f}")
        lines.append(f"  Policy log-prob (mean): {result.dpo.rejected_logprobs.policy_mean:.4f}")
        lines.append(f"  Reference log-prob (sum): {result.dpo.rejected_logprobs.reference_sum:.4f}")
        lines.append(f"  Reference log-prob (mean): {result.dpo.rejected_logprobs.reference_mean:.4f}")
        lines.append(f"  Implicit reward: {result.dpo.rejected_reward:.4f}")
        lines.append("")
        lines.append(f"Margin (chosen - rejected): {result.dpo.margin:.4f}")
        lines.append(f"DPO Loss: {result.dpo.loss:.4f}")

        # GRPO Results
        lines.append("\n--- GRPO (Group Relative Policy Optimization) ---")
        lines.append(f"Group mean reward: {result.grpo.mean_reward:.4f}")
        lines.append(f"Group std reward: {result.grpo.std_reward:.4f}")
        lines.append("")
        lines.append("Per-trace results:")
        for trace_id in result.grpo.trace_ids:
            reward = result.grpo.rewards[trace_id]
            advantage = result.grpo.advantages[trace_id]
            safety = result.grpo.safety_bonuses.get(trace_id, 0)
            safety_str = f" (safety bonus: {safety:.2f})" if safety > 0 else ""
            lines.append(f"  {trace_id}:")
            lines.append(f"    Reward: {reward:.4f}{safety_str}")
            lines.append(f"    Advantage: {advantage:.4f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# For command-line testing
if __name__ == "__main__":
    from config import get_config
    from model_interface import create_model

    config = get_config()

    print("Initializing policy model (pretrained weights)...")
    policy_model = create_model(config.model, use_random_weights=False)

    print("Initializing reference model (RANDOM weights)...")
    reference_model = create_model(
        config.model,
        use_random_weights=True,
        random_seed=config.dpo.reference_model_seed
    )

    # Create test traces
    test_traces = [
        RankedTrace(
            trace_id="test_trace_0",
            text="Let me think about this step by step. First, we need to consider the symptoms. Given the presentation, the answer is B.",
            rank=Rank.BEST
        ),
        RankedTrace(
            trace_id="test_trace_1",
            text="This is a medical question about cardiac issues. After analysis, I believe the answer is B.",
            rank=Rank.MIDDLE
        ),
        RankedTrace(
            trace_id="test_trace_2",
            text="The symptoms suggest a cardiovascular problem. The answer is B.",
            rank=Rank.WORST
        ),
    ]

    test_prompt = "What cardiac abnormality explains these findings? A) MVP B) PFO C) HCM D) VSD"

    print("\nCreating optimizer...")
    optimizer = Optimizer(policy_model, reference_model, config)

    print("\nComputing optimization signals...")
    result = optimizer.compute_optimization_signals(
        sample_id="test_sample",
        prompt=test_prompt,
        traces=test_traces
    )

    print(optimizer.format_results(result))
