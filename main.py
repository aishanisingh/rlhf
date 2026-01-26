"""
trifetch cli

interactive terminal for dpo and grpo signals
"""
import json
import os
import sys
from typing import Dict, List, Tuple

from config import get_config
from model_interface import create_model
from optimizer import Optimizer, RankedTrace, Rank, DPOResult, GRPOResult


# ============== loading ==============

def load_sample(sample_path: str) -> Dict:
    """load sample json"""
    with open(sample_path, 'r') as f:
        data = json.load(f)
    return {
        "question": data.get("Questions") or data.get("question"),
        "answer": data.get("Answer") or data.get("answer"),
        "reasoning": data.get("Reasoning") or data.get("reasoning", "")
    }


def load_traces(cache_path: str) -> List[Dict]:
    """load cached traces"""
    with open(cache_path, 'r') as f:
        data = json.load(f)
    return data.get("verified_traces", [])


def get_available_samples(data_dir: str = ".") -> List[Tuple[str, str, str]]:
    """find samples with traces"""
    samples = []
    cache_dir = os.path.join(data_dir, ".trace_cache")

    for i in range(1, 10):
        sample_path = os.path.join(data_dir, f"sample{i}.json")
        trace_path = os.path.join(cache_dir, f"sample{i}_traces.json")

        if os.path.exists(sample_path) and os.path.exists(trace_path):
            samples.append((f"sample{i}", sample_path, trace_path))

    return samples


# ============== display ==============

def clear_screen():
    """clear screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text: str):
    """header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str):
    """section header"""
    print(f"\n--- {text} ---\n")


def print_trace(idx: int, trace: Dict, show_full: bool = False):
    """display trace"""
    text = trace["text"]
    token_count = trace.get("token_count", len(text.split()))

    print(f"\n[response {idx + 1}] ({token_count} tokens)")
    print("-" * 40)

    if show_full or len(text) < 500:
        print(text)
    else:
        print(text[:500] + "...")
        print(f"\n  (truncated - {len(text)} chars)")
    print()


def print_dpo_results(dpo: DPOResult):
    """show dpo results"""
    print_section("dpo results")

    print(f"  loss:                {dpo.loss:.6f}")
    print(f"  margin:              {dpo.margin:+.6f}")
    print(f"  beta:                {dpo.beta:.4f} (effective: {dpo.effective_beta:.4f})")
    print()
    print(f"  chosen trace:        {dpo.chosen_trace_id}")
    print(f"    policy logprob:    {dpo.chosen_logprobs.policy_mean:.4f}")
    print(f"    reference logprob: {dpo.chosen_logprobs.reference_mean:.4f}")
    print(f"    implicit reward:   {dpo.chosen_reward:+.4f}")
    print()
    print(f"  rejected trace:      {dpo.rejected_trace_id}")
    print(f"    policy logprob:    {dpo.rejected_logprobs.policy_mean:.4f}")
    print(f"    reference logprob: {dpo.rejected_logprobs.reference_mean:.4f}")
    print(f"    implicit reward:   {dpo.rejected_reward:+.4f}")
    print()

    if dpo.margin > 0:
        print("  model already prefers chosen (good)")
    else:
        print("  model prefers rejected (needs training)")


def print_grpo_results(grpo: GRPOResult, rankings: Dict[str, Rank]):
    """show grpo results"""
    print_section("grpo results")

    print(f"  mean reward:   {grpo.mean_reward:.4f}")
    print(f"  std reward:    {grpo.std_reward:.4f}")
    print()
    print("  advantages:")

    for trace_id in grpo.trace_ids:
        rank = rankings.get(trace_id, Rank.MIDDLE)
        reward = grpo.rewards[trace_id]
        advantage = grpo.advantages[trace_id]

        print(f"    {rank.value:6s}: reward={reward:.2f}, advantage={advantage:+.4f}")

    print()
    print("  positive = better than avg")
    print("  negative = worse than avg")


# ============== interactive ==============

def select_sample(samples: List[Tuple[str, str, str]]) -> Tuple[str, str, str]:
    """pick a sample"""
    print_section("samples")

    for i, (sample_id, sample_path, _) in enumerate(samples):
        sample = load_sample(sample_path)
        question_preview = sample["question"][:80].replace("\n", " ")
        print(f"  {i + 1}. {sample_id}: {question_preview}...")

    print()
    while True:
        try:
            choice = input("pick one (1-{}): ".format(len(samples))).strip()
            idx = int(choice) - 1
            if 0 <= idx < len(samples):
                return samples[idx]
        except (ValueError, IndexError):
            pass
        print("invalid, try again")


def get_rankings(traces: List[Dict]) -> Dict[str, Rank]:
    """get user rankings"""
    print_section("rank the traces")
    print("b = best, m = middle, w = worst")
    print("each rank used once\n")

    rankings = {}
    rank_map = {'b': Rank.BEST, 'm': Rank.MIDDLE, 'w': Rank.WORST}
    used_ranks = set()

    for i, trace in enumerate(traces):
        print_trace(i, trace, show_full=True)

        while True:
            choice = input(f"rank for response {i + 1} (b/m/w): ").strip().lower()

            if choice in rank_map:
                rank = rank_map[choice]
                if rank in used_ranks:
                    print(f"  already used {rank.value}")
                else:
                    rankings[trace["trace_id"]] = rank
                    used_ranks.add(rank)
                    print(f"  -> {rank.value}")
                    break
            else:
                print("  use b, m, or w")

    return rankings


def run_interactive(optimizer: Optimizer, samples: List[Tuple[str, str, str]]):
    """main loop"""
    while True:
        clear_screen()
        print_header("trifetch")

        sample_id, sample_path, trace_path = select_sample(samples)
        sample = load_sample(sample_path)
        traces = load_traces(trace_path)

        if len(traces) < 3:
            print(f"\nerror: need 3 traces, found {len(traces)}")
            input("press enter...")
            continue

        traces = traces[:3]

        clear_screen()
        print_header(f"sample: {sample_id}")

        print_section("question")
        print(sample["question"])
        print(f"\nanswer: {sample['answer']}")

        rankings = get_rankings(traces)

        ranked_traces = []
        for trace in traces:
            ranked_traces.append(RankedTrace(
                trace_id=trace["trace_id"],
                text=trace["text"],
                rank=rankings[trace["trace_id"]]
            ))

        print_section("computing...")
        print("this might take a sec\n")

        result = optimizer.compute_optimization_signals(
            sample_id=sample_id,
            prompt=sample["question"],
            traces=ranked_traces
        )

        clear_screen()
        print_header(f"results: {sample_id}")

        print_dpo_results(result.dpo)
        print_grpo_results(result.grpo, rankings)

        print("\n" + "=" * 60)
        choice = input("\nanother one? (y/n): ").strip().lower()
        if choice != 'y':
            break

    print("\nbye!")


# ============== main ==============

def main():
    """entry point"""
    print_header("trifetch")
    print("\nloading models...\n")

    config = get_config()

    samples = get_available_samples(".")
    if not samples:
        print("error: no samples found")
        print("run 'python sampler.py' first")
        sys.exit(1)

    print(f"found {len(samples)} samples\n")

    print("loading policy model...")
    policy_model = create_model(
        config.model,
        use_random_weights=False,
        enable_cache=True,
        cache_file=".logprob_cache_policy.json"
    )
    print(f"  -> {policy_model.get_model_info()}\n")

    print("loading reference model...")
    reference_model = create_model(
        config.model,
        use_random_weights=True,
        random_seed=config.dpo.reference_model_seed,
        enable_cache=True,
        cache_file=".logprob_cache_reference.json"
    )
    print(f"  -> {reference_model.get_model_info()}\n")

    optimizer = Optimizer(policy_model, reference_model, config)

    print("ready!")
    input("\npress enter to start...")

    run_interactive(optimizer, samples)


if __name__ == "__main__":
    main()
