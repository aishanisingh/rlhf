"""
DPO and GRPO optimization logic for TriFetch RLHF Workbench.

This module implements two post-training preference optimization methods:

1. DPO (Direct Preference Optimization)
   - Replaces PPO-based RLHF with a simpler, more stable approach
   - Derives implicit rewards from policy/reference log-prob ratios
   - No separate reward model needed - the policy IS the reward model
   - Based on the Bradley-Terry preference model

2. GRPO (Group Relative Policy Optimization)
   - Extends preference learning to groups of responses (not just pairs)
   - Normalizes advantages relative to the group mean/std
   - Better utilizes data when multiple candidates exist per prompt

Healthcare-specific considerations:
- We rank REASONING TRACES, not just final answers
- All traces shown have correct answers (via rejection sampling)
- This isolates reasoning quality from answer correctness
- Optional safety shaping boosts traces with clinical escalation language
  (e.g., "consult specialist", "emergency", "refer to")

Post-training pipeline context:
  Pretrain -> SFT (supervised fine-tuning) -> Preference Optimization (DPO/GRPO)

This module computes the optimization signals (loss, margin, advantages) that would
be used to update model weights. The math here is exactly what runs during training.
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

    DPO is a post-training method that learns from human preferences without
    training a separate reward model. Key insight: we can derive an implicit
    reward directly from the policy's log-probabilities.

    Theory (Rafailov et al., 2023):
    - The optimal policy under KL-constrained reward maximization has a closed form
    - This lets us reparameterize the reward as: r(x,y) = beta * log(pi/pi_ref)
    - We then optimize the Bradley-Terry preference model directly

    The implicit reward for a completion y given prompt x is:
        r(x, y) = log(pi_policy(y|x)) - log(pi_ref(y|x))

    The loss is: -log(sigmoid(beta * (r_chosen - r_rejected)))

    Why DPO over PPO?
    - Simpler: no reward model, no value function, no clipping heuristics
    - Stable: direct optimization of preference objective
    - Practical: fewer hyperparameters, easier to debug and interpret

    Healthcare note:
    - chosen_trace and rejected_trace both have CORRECT ANSWERS
    - we're comparing REASONING QUALITY, not answer correctness
    - this matters because a model that gets right answers for wrong reasons
      will fail on edge cases where the shortcut doesn't apply

    Args:
        prompt: The input prompt (medical question)
        chosen_trace: The trace ranked as Best (better reasoning)
        rejected_trace: The trace ranked as Worst (worse reasoning)
        policy_model: Policy model with pretrained weights
        reference_model: Reference model (baseline for KL constraint)
        config: DPO configuration (beta, log-prob mode, etc.)

    Returns:
        DPOResult with loss, margin, implicit rewards, and log-probs
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

    GRPO extends preference optimization beyond pairwise comparisons.
    Instead of just best-vs-worst, we consider all responses together
    and compute advantages relative to the group.

    Theory:
    - Assign rewards based on rank: best=1.0, middle=0.5, worst=0.0
    - Normalize to get advantages: (reward - mean) / std
    - This is similar to advantage estimation in policy gradient methods
    - The std normalization keeps gradient magnitudes consistent

    Why group-relative?
    - Uses more information than pairwise (DPO only uses best vs worst)
    - The middle trace contributes to learning, not just extremes
    - Variance reduction through normalization improves training stability

    Healthcare-specific: Safety Shaping
    - Optional bonus for traces containing clinical escalation language
    - Keywords like "consult specialist", "emergency", "refer to"
    - In medical AI, we WANT models that suggest escalation when uncertain
    - This rewards clinically cautious reasoning

    Advantage interpretation:
    - Positive advantage: trace is better than group average -> reinforce
    - Negative advantage: trace is worse than group average -> discourage
    - The magnitude indicates how much better/worse

    Args:
        traces: List of all three ranked traces (Best, Middle, Worst)
        config: GRPO configuration (rewards, safety shaping, etc.)

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

        # Healthcare safety shaping:
        # In medical AI, false negatives (missing a serious condition) are worse
        # than false positives (unnecessary referral). We want models that err
        # on the side of caution - suggesting specialist consultation, emergency
        # care, or further testing when there's uncertainty.
        #
        # This bonus rewards traces that include clinical escalation language,
        # even if their base reasoning rank is lower. A trace that says
        # "this could be X, but consult cardiology to rule out Y" shows
        # appropriate clinical caution.
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

    This class computes the optimization signals that would be used in
    post-training to align a model with human preferences. The signals
    computed here are exactly what would flow through the backward pass
    during actual training.

    Design philosophy:
    - Pragmatic: focuses on the math that matters (loss computation)
    - Extensible: clean separation between DPO and GRPO
    - Healthcare-aware: supports safety shaping for clinical contexts

    In a full training loop, you would:
    1. Compute these signals for a batch of samples
    2. Backpropagate the DPO loss through the policy model
    3. Update weights with your optimizer (AdamW, etc.)

    We compute signals without training to:
    - Demonstrate understanding of the underlying math
    - Allow exploration without GPU requirements
    - Focus on preference learning logic, not training infrastructure
    """

    def __init__(
        self,
        policy_model: ModelInterface,
        reference_model: ModelInterface,
        config: Config
    ):
        """
        Initialize optimizer with models and configuration.

        The reference model is crucial for DPO - it provides the baseline
        distribution that we compute KL divergence against. In standard DPO,
        this is the SFT model (post supervised fine-tuning, pre preference
        optimization). Here we use random weights to simulate this baseline.

        Args:
            policy_model: Policy model with pretrained weights
            reference_model: Reference model (baseline for KL constraint)
            config: Full configuration including DPO and GRPO settings
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
