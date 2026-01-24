"""
Streamlit UI for TriFetch Online RLHF Workbench.

Provides interactive interface for clinicians to rank reasoning traces
and compute optimization signals using DPO and GRPO.
"""
import streamlit as st
import json
import os
from typing import Dict, List, Optional

from config import get_config, Config
from sampler import load_all_samples, MedicalQASample, VerifiedTraceSet, TraceCache, ReasoningTrace
from optimizer import Optimizer, RankedTrace, Rank, OptimizationResult
from model_interface import create_model, ModelInterface


# Page configuration
st.set_page_config(
    page_title="TriFetch RLHF Workbench",
    page_icon="🏥",
    layout="wide"
)


@st.cache_resource
def load_models(config: Config):
    """Load and cache policy and reference models."""
    with st.spinner("Loading policy model (pretrained weights)..."):
        policy_model = create_model(config.model, use_random_weights=False)

    # Standard DPO: reference model uses same pretrained weights as initial policy
    use_random = not config.dpo.use_pretrained_reference
    ref_label = "pretrained" if config.dpo.use_pretrained_reference else "random"
    with st.spinner(f"Loading reference model ({ref_label} weights)..."):
        reference_model = create_model(
            config.model,
            use_random_weights=use_random,
            random_seed=config.dpo.reference_model_seed if use_random else None,
            cache_file=".logprob_cache_ref.json"
        )

    return policy_model, reference_model


@st.cache_data
def load_cached_traces(_config: Config) -> Dict[str, Dict]:
    """Load pre-generated traces from cache."""
    cache = TraceCache(_config.sampler.cache_dir)
    traces_by_sample = {}

    for filename in _config.sample_files:
        sample_id = os.path.splitext(filename)[0]
        cached = cache.load(sample_id)
        if cached and len(cached.get("verified_traces", [])) >= _config.sampler.required_traces:
            traces_by_sample[sample_id] = cached

    return traces_by_sample


def load_samples_data(config: Config) -> Dict[str, MedicalQASample]:
    """Load sample data from JSON files."""
    samples = load_all_samples(config)
    return {s.sample_id: s for s in samples}


def render_trace_card(trace: Dict, index: int, selected_rank: Optional[str] = None) -> str:
    """Render a trace card with ranking selector."""
    return trace.get("text", "")[:500] + "..." if len(trace.get("text", "")) > 500 else trace.get("text", "")


def main():
    st.title("🏥 TriFetch Online RLHF Workbench")
    st.markdown("""
    **Medical AI Post-Training Control Room**

    This workbench allows clinicians to rank AI reasoning traces and compute
    optimization signals for model improvement using DPO and GRPO methods.
    """)

    # Load configuration
    config = get_config()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        st.subheader("DPO Settings")
        beta = st.slider("Beta (KL penalty)", 0.01, 1.0, config.dpo.beta, 0.01)
        config.dpo.beta = beta

        use_length_scaling = st.checkbox(
            "Token-length-aware beta scaling",
            config.dpo.use_length_scaling
        )
        config.dpo.use_length_scaling = use_length_scaling

        log_prob_mode = st.selectbox(
            "Log-prob mode",
            ["length_normalized", "sum"],
            index=0 if config.dpo.log_prob_mode.value == "length_normalized" else 1
        )
        from config import LogProbMode
        config.dpo.log_prob_mode = LogProbMode(log_prob_mode)
        config.grpo.log_prob_mode = LogProbMode(log_prob_mode)

        st.subheader("GRPO Settings")
        use_safety_shaping = st.checkbox(
            "Healthcare safety shaping",
            config.grpo.use_safety_shaping,
            help="Boost traces with escalation language (consult, emergency, etc.)"
        )
        config.grpo.use_safety_shaping = use_safety_shaping

        use_exp_decay = st.checkbox(
            "Exponential decay rewards",
            config.grpo.use_exponential_decay
        )
        config.grpo.use_exponential_decay = use_exp_decay

        st.divider()
        st.caption("Model: " + config.model.local_model_name)

    # Load models
    try:
        policy_model, reference_model = load_models(config)
        models_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure you have the required model dependencies installed.")
        models_loaded = False

    # Load sample data
    try:
        samples = load_samples_data(config)
    except Exception as e:
        st.error(f"Error loading samples: {e}")
        return

    # Load cached traces
    cached_traces = load_cached_traces(config)

    if not cached_traces:
        st.warning("""
        **No pre-generated traces found.**

        Please run the sampler first to generate traces:
        ```bash
        python sampler.py
        ```

        This will generate 3 distinct verified reasoning traces for each sample.
        """)

        # Show option to use demo traces
        if st.button("Use Demo Traces (for testing)"):
            # Create demo traces for testing the UI
            for sample_id, sample in samples.items():
                demo_traces = [
                    {
                        "trace_id": f"{sample_id}_demo_0",
                        "text": f"Let me analyze this medical case step by step. Looking at the symptoms presented, I need to consider the differential diagnosis carefully. After evaluating all options, based on the clinical presentation, the answer is {sample.answer}.",
                        "extracted_answer": sample.answer,
                        "is_correct": True,
                        "normalized_hash": "demo_hash_0",
                        "token_count": 50
                    },
                    {
                        "trace_id": f"{sample_id}_demo_1",
                        "text": f"This is an interesting clinical scenario. The key findings point us toward a specific diagnosis. Considering the pathophysiology involved, I would recommend further evaluation. The answer is {sample.answer}.",
                        "extracted_answer": sample.answer,
                        "is_correct": True,
                        "normalized_hash": "demo_hash_1",
                        "token_count": 45
                    },
                    {
                        "trace_id": f"{sample_id}_demo_2",
                        "text": f"Based on the clinical presentation and the patient's history, we should consider emergency consultation. The symptoms suggest the answer is {sample.answer}.",
                        "extracted_answer": sample.answer,
                        "is_correct": True,
                        "normalized_hash": "demo_hash_2",
                        "token_count": 35
                    },
                ]
                cached_traces[sample_id] = {"verified_traces": demo_traces}
            st.rerun()

    if not cached_traces:
        return

    # Sample selector
    st.subheader("Select Medical Case")
    available_samples = [s for s in samples.keys() if s in cached_traces]

    if not available_samples:
        st.error("No samples have traces generated. Please run sampler.py first.")
        return

    selected_sample_id = st.selectbox(
        "Choose a medical QA sample:",
        available_samples,
        format_func=lambda x: f"{x} - {samples[x].answer}"
    )

    sample = samples[selected_sample_id]
    traces_data = cached_traces[selected_sample_id]["verified_traces"][:3]

    # Display the medical question
    st.subheader("Medical Question")
    with st.expander("View Full Question", expanded=True):
        st.markdown(sample.question)
        st.info(f"**Ground Truth Answer: {sample.answer}**")

    # Display traces for ranking
    st.subheader("Rank the Reasoning Traces")
    st.markdown("""
    As a clinician, please rank these three AI-generated reasoning traces.
    Each trace arrives at the correct answer but uses different reasoning paths.
    """)

    # Initialize session state for rankings
    if "rankings" not in st.session_state:
        st.session_state.rankings = {}

    if selected_sample_id not in st.session_state.rankings:
        st.session_state.rankings[selected_sample_id] = {
            traces_data[0]["trace_id"]: None,
            traces_data[1]["trace_id"]: None,
            traces_data[2]["trace_id"]: None,
        }

    # Display traces in columns
    cols = st.columns(3)
    rank_options = ["Best", "Middle", "Worst"]

    for i, (col, trace) in enumerate(zip(cols, traces_data)):
        with col:
            st.markdown(f"**Trace {i+1}**")

            # Show trace text in a container
            with st.container(border=True):
                trace_text = trace.get("text", "")
                if len(trace_text) > 1000:
                    st.markdown(trace_text[:1000] + "...")
                    with st.expander("Show full trace"):
                        st.markdown(trace_text)
                else:
                    st.markdown(trace_text)

            # Ranking selector
            trace_id = trace["trace_id"]
            current_rank = st.session_state.rankings[selected_sample_id].get(trace_id)

            rank = st.selectbox(
                f"Rank for Trace {i+1}",
                ["(Select)", "Best", "Middle", "Worst"],
                index=0 if current_rank is None else rank_options.index(current_rank) + 1,
                key=f"rank_{trace_id}"
            )

            if rank != "(Select)":
                st.session_state.rankings[selected_sample_id][trace_id] = rank

    # Validate rankings
    current_rankings = st.session_state.rankings[selected_sample_id]
    assigned_ranks = [r for r in current_rankings.values() if r is not None]

    rankings_valid = (
        len(assigned_ranks) == 3 and
        len(set(assigned_ranks)) == 3 and
        set(assigned_ranks) == {"Best", "Middle", "Worst"}
    )

    if assigned_ranks and not rankings_valid:
        st.warning("Please assign exactly one trace to each rank: Best, Middle, and Worst.")

    # Update Model button
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        update_button = st.button(
            "🔄 Update Model (Compute Optimization Signals)",
            type="primary",
            disabled=not (rankings_valid and models_loaded),
            use_container_width=True
        )

    if update_button and rankings_valid and models_loaded:
        # Build RankedTrace objects
        rank_map = {"Best": Rank.BEST, "Middle": Rank.MIDDLE, "Worst": Rank.WORST}
        ranked_traces = []

        for trace in traces_data:
            trace_id = trace["trace_id"]
            rank_str = current_rankings[trace_id]
            ranked_traces.append(RankedTrace(
                trace_id=trace_id,
                text=trace["text"],
                rank=rank_map[rank_str]
            ))

        # Create optimizer and compute signals
        with st.spinner("Computing optimization signals..."):
            optimizer = Optimizer(policy_model, reference_model, config)

            try:
                result = optimizer.compute_optimization_signals(
                    sample_id=selected_sample_id,
                    prompt=sample.question,
                    traces=ranked_traces
                )

                # Display results
                st.success("Optimization signals computed successfully!")

                # Results in tabs
                tab1, tab2, tab3 = st.tabs(["DPO Results", "GRPO Results", "Raw Output"])

                with tab1:
                    st.subheader("Direct Preference Optimization (DPO)")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("DPO Loss", f"{result.dpo.loss:.4f}")
                        st.metric("Margin (chosen - rejected)", f"{result.dpo.margin:.4f}")

                    with col2:
                        st.metric("Beta", f"{result.dpo.effective_beta:.4f}")
                        st.metric("Log-prob mode", result.dpo.log_prob_mode.value)

                    st.markdown("---")

                    # Chosen trace details
                    st.markdown("**Chosen (Best) Trace**")
                    st.markdown(f"- Trace ID: `{result.dpo.chosen_trace_id}`")
                    st.markdown(f"- Policy log-prob (sum): `{result.dpo.chosen_logprobs.policy_sum:.4f}`")
                    st.markdown(f"- Policy log-prob (mean): `{result.dpo.chosen_logprobs.policy_mean:.4f}`")
                    st.markdown(f"- Reference log-prob (sum): `{result.dpo.chosen_logprobs.reference_sum:.4f}`")
                    st.markdown(f"- Reference log-prob (mean): `{result.dpo.chosen_logprobs.reference_mean:.4f}`")
                    st.markdown(f"- **Implicit Reward**: `{result.dpo.chosen_reward:.4f}`")

                    st.markdown("---")

                    # Rejected trace details
                    st.markdown("**Rejected (Worst) Trace**")
                    st.markdown(f"- Trace ID: `{result.dpo.rejected_trace_id}`")
                    st.markdown(f"- Policy log-prob (sum): `{result.dpo.rejected_logprobs.policy_sum:.4f}`")
                    st.markdown(f"- Policy log-prob (mean): `{result.dpo.rejected_logprobs.policy_mean:.4f}`")
                    st.markdown(f"- Reference log-prob (sum): `{result.dpo.rejected_logprobs.reference_sum:.4f}`")
                    st.markdown(f"- Reference log-prob (mean): `{result.dpo.rejected_logprobs.reference_mean:.4f}`")
                    st.markdown(f"- **Implicit Reward**: `{result.dpo.rejected_reward:.4f}`")

                    st.markdown("---")
                    st.info("""
                    **DPO Interpretation**: The margin is `chosen_reward - rejected_reward`.
                    Positive margin = model correctly prefers chosen trace (low loss).
                    Negative margin = model incorrectly prefers rejected trace (high loss).
                    """)

                with tab2:
                    st.subheader("Group Relative Policy Optimization (GRPO)")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Group Mean Reward", f"{result.grpo.mean_reward:.4f}")

                    with col2:
                        st.metric("Group Std Reward", f"{result.grpo.std_reward:.4f}")

                    st.markdown("---")

                    # Per-trace advantages
                    st.markdown("**Per-Trace Results**")

                    for trace_id in result.grpo.trace_ids:
                        reward = result.grpo.rewards[trace_id]
                        advantage = result.grpo.advantages[trace_id]
                        safety_bonus = result.grpo.safety_bonuses.get(trace_id, 0)

                        rank_str = current_rankings.get(trace_id, "Unknown")

                        with st.container(border=True):
                            st.markdown(f"**{trace_id}** ({rank_str})")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Reward", f"{reward:.4f}")
                            col2.metric("Advantage", f"{advantage:.4f}")
                            if safety_bonus > 0:
                                col3.metric("Safety Bonus", f"+{safety_bonus:.2f}")

                    if config.grpo.use_safety_shaping:
                        st.info("""
                        **Healthcare Safety Shaping**: Traces containing escalation language
                        (consult, emergency, urgent, etc.) receive a bonus reward to encourage
                        appropriate clinical caution.
                        """)

                with tab3:
                    st.subheader("Raw Output")
                    st.code(optimizer.format_results(result))

            except Exception as e:
                st.error(f"Error computing optimization signals: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
