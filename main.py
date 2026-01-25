"""
TriFetch RLHF Workbench - Local CLI Interface

Interactive terminal application for computing DPO and GRPO optimization signals
using real model log-probabilities.
"""
import json
import os
import sys
from typing import Dict, List, Tuple

from config import get_config
from model_interface import create_model
from optimizer import Optimizer, RankedTrace, Rank, DPOResult, GRPOResult


# ============== Data Loading ==============

def load_sample(sample_path: str) -> Dict:
    """Load a medical QA sample from JSON."""
    with open(sample_path, 'r') as f:
        data = json.load(f)
    return {
        "question": data.get("Questions") or data.get("question"),
        "answer": data.get("Answer") or data.get("answer"),
        "reasoning": data.get("Reasoning") or data.get("reasoning", "")
    }


def load_traces(cache_path: str) -> List[Dict]:
    """Load pre-generated traces from cache."""
    with open(cache_path, 'r') as f:
        data = json.load(f)
    return data.get("verified_traces", [])


def get_available_samples(data_dir: str = ".") -> List[Tuple[str, str, str]]:
    """Find all samples with cached traces.

    Returns list of (sample_id, sample_path, trace_path) tuples.
    """
    samples = []
    cache_dir = os.path.join(data_dir, ".trace_cache")

    for i in range(1, 10):
        sample_path = os.path.join(data_dir, f"sample{i}.json")
        trace_path = os.path.join(cache_dir, f"sample{i}_traces.json")

        if os.path.exists(sample_path) and os.path.exists(trace_path):
            samples.append((f"sample{i}", sample_path, trace_path))

    return samples


# ============== Display Functions ==============

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str):
    """Print a section header."""
    print(f"\n--- {text} ---\n")


def print_trace(idx: int, trace: Dict, show_full: bool = False):
    """Display a trace with formatting."""
    text = trace["text"]
    token_count = trace.get("token_count", len(text.split()))

    print(f"\n[Response {idx + 1}] ({token_count} tokens)")
    print("-" * 40)

    if show_full or len(text) < 500:
        print(text)
    else:
        # truncate for display
        print(text[:500] + "...")
        print(f"\n  (truncated - {len(text)} chars total)")
    print()


def print_dpo_results(dpo: DPOResult):
    """Display DPO computation results."""
    print_section("DPO Results (Direct Preference Optimization)")

    print(f"  Loss:                {dpo.loss:.6f}")
    print(f"  Margin:              {dpo.margin:+.6f}")
    print(f"  Beta:                {dpo.beta:.4f} (effective: {dpo.effective_beta:.4f})")
    print()
    print(f"  Chosen trace:        {dpo.chosen_trace_id}")
    print(f"    Policy log-prob:   {dpo.chosen_logprobs.policy_mean:.4f} (mean)")
    print(f"    Reference log-prob:{dpo.chosen_logprobs.reference_mean:.4f} (mean)")
    print(f"    Implicit reward:   {dpo.chosen_reward:+.4f}")
    print()
    print(f"  Rejected trace:      {dpo.rejected_trace_id}")
    print(f"    Policy log-prob:   {dpo.rejected_logprobs.policy_mean:.4f} (mean)")
    print(f"    Reference log-prob:{dpo.rejected_logprobs.reference_mean:.4f} (mean)")
    print(f"    Implicit reward:   {dpo.rejected_reward:+.4f}")
    print()

    if dpo.margin > 0:
        print("  Interpretation: Model already prefers the chosen response.")
    else:
        print("  Interpretation: Model prefers the rejected response - training needed.")


def print_grpo_results(grpo: GRPOResult, rankings: Dict[str, Rank]):
    """Display GRPO computation results."""
    print_section("GRPO Results (Group Relative Policy Optimization)")

    print(f"  Mean reward:   {grpo.mean_reward:.4f}")
    print(f"  Std reward:    {grpo.std_reward:.4f}")
    print()
    print("  Per-trace advantages:")

    for trace_id in grpo.trace_ids:
        rank = rankings.get(trace_id, Rank.MIDDLE)
        reward = grpo.rewards[trace_id]
        advantage = grpo.advantages[trace_id]

        print(f"    {rank.value:6s}: reward={reward:.2f}, advantage={advantage:+.4f}")

    print()
    print("  Interpretation:")
    print("    Positive advantage = better than group average")
    print("    Negative advantage = worse than group average")


# ============== Interactive CLI ==============

def select_sample(samples: List[Tuple[str, str, str]]) -> Tuple[str, str, str]:
    """Let user select a sample."""
    print_section("Available Samples")

    for i, (sample_id, sample_path, _) in enumerate(samples):
        sample = load_sample(sample_path)
        question_preview = sample["question"][:80].replace("\n", " ")
        print(f"  {i + 1}. {sample_id}: {question_preview}...")

    print()
    while True:
        try:
            choice = input("Select sample (1-{}): ".format(len(samples))).strip()
            idx = int(choice) - 1
            if 0 <= idx < len(samples):
                return samples[idx]
        except (ValueError, IndexError):
            pass
        print("Invalid selection. Try again.")


def get_rankings(traces: List[Dict]) -> Dict[str, Rank]:
    """Get user rankings for traces."""
    print_section("Rank the Responses")
    print("Assign each response a rank: Best (b), Middle (m), or Worst (w)")
    print("Each rank must be used exactly once.\n")

    rankings = {}
    rank_map = {'b': Rank.BEST, 'm': Rank.MIDDLE, 'w': Rank.WORST}
    used_ranks = set()

    for i, trace in enumerate(traces):
        print_trace(i, trace, show_full=True)

        while True:
            choice = input(f"Rank for Response {i + 1} (b/m/w): ").strip().lower()

            if choice in rank_map:
                rank = rank_map[choice]
                if rank in used_ranks:
                    print(f"  '{rank.value}' already assigned. Choose another.")
                else:
                    rankings[trace["trace_id"]] = rank
                    used_ranks.add(rank)
                    print(f"  -> {rank.value}")
                    break
            else:
                print("  Invalid input. Use 'b' for Best, 'm' for Middle, 'w' for Worst.")

    return rankings


def run_interactive(optimizer: Optimizer, samples: List[Tuple[str, str, str]]):
    """Run the interactive CLI loop."""
    while True:
        clear_screen()
        print_header("TriFetch RLHF Workbench")

        # select sample
        sample_id, sample_path, trace_path = select_sample(samples)
        sample = load_sample(sample_path)
        traces = load_traces(trace_path)

        if len(traces) < 3:
            print(f"\nError: Need at least 3 traces, found {len(traces)}")
            input("Press Enter to continue...")
            continue

        # limit to 3 traces
        traces = traces[:3]

        clear_screen()
        print_header(f"Sample: {sample_id}")

        # show question
        print_section("Question")
        print(sample["question"])
        print(f"\nCorrect Answer: {sample['answer']}")

        # get rankings
        rankings = get_rankings(traces)

        # convert to RankedTrace objects
        ranked_traces = []
        for trace in traces:
            ranked_traces.append(RankedTrace(
                trace_id=trace["trace_id"],
                text=trace["text"],
                rank=rankings[trace["trace_id"]]
            ))

        # compute optimization signals
        print_section("Computing Optimization Signals")
        print("This may take a moment as we compute log-probabilities...")
        print()

        result = optimizer.compute_optimization_signals(
            sample_id=sample_id,
            prompt=sample["question"],
            traces=ranked_traces
        )

        # display results
        clear_screen()
        print_header(f"Results: {sample_id}")

        print_dpo_results(result.dpo)
        print_grpo_results(result.grpo, rankings)

        print("\n" + "=" * 60)
        choice = input("\nAnother sample? (y/n): ").strip().lower()
        if choice != 'y':
            break

    print("\nGoodbye!")


# ============== Main ==============

def main():
    """Main entry point."""
    print_header("TriFetch RLHF Workbench")
    print("\nInitializing models...\n")

    # get config
    config = get_config()

    # check for available samples
    samples = get_available_samples(".")
    if not samples:
        print("Error: No samples with cached traces found.")
        print("Run 'python sampler.py' first to generate traces.")
        sys.exit(1)

    print(f"Found {len(samples)} samples with cached traces.\n")

    # load policy model (pretrained)
    print("Loading policy model (pretrained weights)...")
    policy_model = create_model(
        config.model,
        use_random_weights=False,
        enable_cache=True,
        cache_file=".logprob_cache_policy.json"
    )
    print(f"  -> {policy_model.get_model_info()}\n")

    # load reference model (random weights)
    print("Loading reference model (random weights)...")
    reference_model = create_model(
        config.model,
        use_random_weights=True,
        random_seed=config.dpo.reference_model_seed,
        enable_cache=True,
        cache_file=".logprob_cache_reference.json"
    )
    print(f"  -> {reference_model.get_model_info()}\n")

    # create optimizer
    optimizer = Optimizer(policy_model, reference_model, config)

    print("Models loaded successfully!")
    input("\nPress Enter to start...")

    # run interactive loop
    run_interactive(optimizer, samples)


if __name__ == "__main__":
    main()
