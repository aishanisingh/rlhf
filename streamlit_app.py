"""
Streamlit UI for TriFetch Online RLHF Workbench.
Lightweight version for cloud deployment.
"""
import streamlit as st
import json
import math
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

# Page configuration
st.set_page_config(
    page_title="RLHF Workbench",
    page_icon="🏥",
    layout="wide"
)

# ============== Embedded Demo Data ==============

SAMPLES = {
    "sample1": {
        "question": """Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?

A) Mitral Valve Prolapse
B) Patent Foramen Ovale
C) Hypertrophic Cardiomyopathy
D) Ventricular Septal Defect""",
        "answer": "B"
    },
    "sample2": {
        "question": """A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. Her pulse is 110/min, respirations 22/min, and blood pressure 90/65 mm Hg. There is a 5-cm deep stab wound at the upper border of the 8th rib in the left midaxillary line. Which anatomical structure is most likely to be injured?

A) Left atrium of the heart
B) Lower lobe of the left lung
C) Spleen
D) Left lobe of the liver""",
        "answer": "B"
    },
    "sample3": {
        "question": """A patient presents with progressive gait disturbances, tremors, and speech difficulties. Genetic testing confirms the presence of GAA trinucleotide repeat expansions. Which chromosome is most commonly associated with the mutated gene in this condition?

A) Chromosome 4
B) Chromosome 6
C) Chromosome 9
D) Chromosome X""",
        "answer": "C"
    },
    "sample4": {
        "question": """A 25-year-old male presents with high-grade fever and hypotension. Laboratory results show hemoglobin 5 g/dL, total leukocyte count 9000/mm3, and a differential count of 2% polymorphs, 96% lymphocytes, and 2% eosinophils. Which of the following treatment options should be avoided in this clinical scenario?

A) Intravenous fluid resuscitation
B) Packed red blood cell transfusion
C) Oral ciprofloxacin
D) Intravenous broad-spectrum antibiotics""",
        "answer": "C"
    },
    "sample5": {
        "question": """A 32-year-old man presents with a severe headache in the left forehead and eye that wakes him from sleep. He has a history of a recent sinus infection and type 1 diabetes. Imaging reveals thrombosis of a sinus located above the sella turcica. Which of the following findings would most likely also be seen in this patient?

A) Anosmia
B) Mandibular pain
C) Ophthalmoplegia
D) Vertigo
E) Vision loss""",
        "answer": "C"
    }
}

TRACES = {
    "sample1": [
        {
            "id": "trace_0",
            "text": """Let me analyze this case systematically. The patient presents with sudden weakness in the left arm and leg, which suggests a neurological event - likely a stroke affecting the right side of the brain.

Key observation: The patient has a swollen, tender right lower leg after long-distance travel. This is classic for deep vein thrombosis (DVT).

Now, the critical question is: how could a venous clot cause an arterial stroke? Normally, venous blood goes through the lungs where clots would be filtered. However, if there's a right-to-left cardiac shunt, a clot could bypass the lungs entirely.

A Patent Foramen Ovale (PFO) is exactly such a shunt - a hole between the right and left atria that persists from fetal circulation. It's present in about 25% of adults.

This explains the paradoxical embolism: DVT clot → right atrium → through PFO → left atrium → brain → stroke.

The answer is B."""
        },
        {
            "id": "trace_1",
            "text": """This patient has neurological symptoms (left-sided weakness) combined with signs of DVT (swollen leg after travel).

When we see stroke symptoms in a young patient with DVT, we should think about paradoxical embolism. This happens when a venous clot crosses to the arterial circulation through a cardiac defect.

Looking at the options:
- Mitral Valve Prolapse: causes regurgitation, not shunting
- Patent Foramen Ovale: allows right-to-left shunting - this fits!
- Hypertrophic Cardiomyopathy: causes outflow obstruction
- VSD: usually left-to-right shunt in adults

The answer is B."""
        },
        {
            "id": "trace_2",
            "text": """The patient has weakness on one side and a swollen leg. The leg swelling after travel suggests a clot.

I need to find a heart problem that connects these. A hole in the heart could let a clot go to the brain.

Patent Foramen Ovale is a hole between atria.

The answer is B."""
        }
    ],
    "sample2": [
        {
            "id": "trace_0",
            "text": """This is a trauma case requiring careful anatomical analysis. Let me work through this systematically.

The wound is at the upper border of the 8th rib in the left midaxillary line, with 5cm depth. I need to consider what structures lie in this region.

Anatomical considerations:
- The 8th rib at the midaxillary line is at approximately the level of T8-T9
- This is below the cardiac silhouette but still within the thoracic cavity
- The left hemidiaphragm attaches around ribs 7-10 laterally
- The lower lobe of the left lung extends to about the 10th rib in the midaxillary line

Given the location and depth:
- The heart (left atrium) is too medial and superior
- The spleen is more posterior and inferior
- The liver is on the right side
- The lower lobe of the left lung is directly in the path

The patient's vitals (tachycardia, hypotension) suggest hemothorax or pneumothorax, consistent with lung injury.

The answer is B."""
        },
        {
            "id": "trace_1",
            "text": """A stab wound at the 8th rib, left midaxillary line, 5cm deep. The patient is hypotensive and tachycardic.

At this level on the left side, the main structures are:
- Lower lobe of left lung
- Possibly the diaphragm
- The spleen is nearby but more posterior

The clinical picture with unstable vitals suggests bleeding into the chest, most likely from the lung.

The left atrium is too high and medial. The liver is on the wrong side.

The answer is B."""
        },
        {
            "id": "trace_2",
            "text": """Stab wound on the left side of the chest at rib 8. The patient's blood pressure is low.

At this location, the lung would be hit. The lower lobe of the left lung is in this area.

The answer is B."""
        }
    ],
    "sample3": [
        {
            "id": "trace_0",
            "text": """The clinical presentation of progressive gait disturbances, tremors, and speech difficulties points toward a neurodegenerative condition affecting the cerebellum and its connections.

The genetic finding of GAA trinucleotide repeat expansions is the definitive clue. This pattern is pathognomonic for Friedreich's ataxia.

Let me recall the genetics:
- Friedreich's ataxia is caused by mutations in the FXN gene
- The FXN gene encodes frataxin, a mitochondrial protein
- The gene is located on chromosome 9q21

Let me verify by excluding other options:
- Chromosome 4: Huntington disease (CAG repeats) - different presentation
- Chromosome 6: Not associated with major trinucleotide repeat disorders
- Chromosome X: Fragile X syndrome (CGG repeats) - different presentation

The FXN gene on chromosome 9 is the correct answer.

The answer is C."""
        },
        {
            "id": "trace_1",
            "text": """GAA trinucleotide repeats are characteristic of Friedreich's ataxia. This autosomal recessive condition affects the nervous system and heart.

The responsible gene is FXN (frataxin), and I need to remember which chromosome it's on.

Friedreich's ataxia is associated with chromosome 9 - specifically 9q21 where the FXN gene is located.

The other chromosomes listed are associated with different conditions (Huntington's is on 4, Fragile X involves the X chromosome).

The answer is C."""
        },
        {
            "id": "trace_2",
            "text": """GAA repeats cause Friedreich's ataxia. The gene for this is on chromosome 9.

The answer is C."""
        }
    ],
    "sample4": [
        {
            "id": "trace_0",
            "text": """This is a critically ill patient with fever, hypotension, and severely abnormal labs. Let me analyze systematically.

Key findings:
- Severe anemia: Hb 5 g/dL (critical)
- Abnormal differential: 2% neutrophils (profound neutropenia), 96% lymphocytes
- Signs of sepsis/shock: fever, hypotension

The differential with only 2% polymorphs indicates severe neutropenia. Combined with fever and hypotension, this is febrile neutropenia with septic shock - a medical emergency.

Management priorities:
- IV fluid resuscitation: Essential for shock - should be given
- Packed RBC transfusion: Critical for Hb of 5 - should be given
- IV broad-spectrum antibiotics: Standard of care for febrile neutropenia - should be given

What should be AVOIDED?
- Oral ciprofloxacin: In septic shock, GI absorption is unreliable due to poor perfusion. IV antibiotics are mandatory. Oral route is inadequate for this acuity.

The answer is C."""
        },
        {
            "id": "trace_1",
            "text": """This patient has febrile neutropenia (2% polymorphs) with signs of septic shock.

The treatment needs to be aggressive:
- Fluids for hypotension - yes
- Blood transfusion for severe anemia - yes
- Antibiotics - yes, but must be IV

Oral ciprofloxacin would be inappropriate because:
1. The patient is in shock with unreliable GI absorption
2. Febrile neutropenia requires IV antibiotics
3. Oral medications are too slow-acting for this emergency

The answer is C."""
        },
        {
            "id": "trace_2",
            "text": """The patient has low neutrophils and signs of infection with shock.

You can't give oral antibiotics to someone in septic shock. They need IV antibiotics.

The answer is C."""
        }
    ],
    "sample5": [
        {
            "id": "trace_0",
            "text": """This case requires careful neuroanatomical reasoning. Let me analyze the key features.

Clinical presentation:
- Severe headache in left forehead and eye, waking from sleep
- History of recent sinus infection
- Type 1 diabetes (increases infection risk)
- Imaging shows sinus thrombosis ABOVE the sella turcica

The location "above the sella turcica" is critical - this describes the cavernous sinus. Cavernous sinus thrombosis (CST) is a serious complication of facial/sinus infections.

Structures within the cavernous sinus:
- CN III (oculomotor)
- CN IV (trochlear)
- CN VI (abducens)
- CN V1 and V2 (trigeminal branches)
- Internal carotid artery

When these cranial nerves are affected by inflammation/thrombosis, the result is ophthalmoplegia - inability to move the eye properly.

Let me exclude other options:
- Anosmia: CN I is not in cavernous sinus
- Mandibular pain: V3 is not in cavernous sinus
- Vertigo: CN VIII is not in cavernous sinus
- Vision loss: Optic nerve is not in cavernous sinus

The answer is C."""
        },
        {
            "id": "trace_1",
            "text": """The imaging shows thrombosis above the sella turcica - this is the cavernous sinus.

The cavernous sinus contains cranial nerves III, IV, VI (eye movement) and branches of V.

With cavernous sinus thrombosis, these nerves get compressed, causing ophthalmoplegia (paralysis of eye movements).

The olfactory nerve and vestibular system are not involved. The optic nerve runs separately.

The answer is C."""
        },
        {
            "id": "trace_2",
            "text": """Thrombosis above sella turcica means cavernous sinus problem. Eye movement nerves are in the cavernous sinus.

The answer is C."""
        }
    ]
}


# ============== Optimization Logic ==============

class Rank(Enum):
    BEST = "best"
    MIDDLE = "middle"
    WORST = "worst"


def stable_log_sigmoid(x: float) -> float:
    """Numerically stable log(sigmoid(x))."""
    if x >= 0:
        return -math.log(1 + math.exp(-x) + 1e-8)
    else:
        return x - math.log(1 + math.exp(x) + 1e-8)


def compute_dpo_signals(chosen_text: str, rejected_text: str, beta: float = 0.1):
    """Compute DPO optimization signals (simulated log-probs for demo)."""
    # Simulate log-probs based on text length and complexity
    # In production, these would come from actual model inference

    def simulate_logprob(text: str, is_policy: bool) -> float:
        base = -2.0 * len(text.split())
        # Policy model slightly prefers longer, more detailed responses
        if is_policy:
            base += len(text.split()) * 0.1
        return base

    # Policy and reference log-probs
    chosen_policy_lp = simulate_logprob(chosen_text, True)
    chosen_ref_lp = simulate_logprob(chosen_text, False)
    rejected_policy_lp = simulate_logprob(rejected_text, True)
    rejected_ref_lp = simulate_logprob(rejected_text, False)

    # Implicit rewards
    chosen_reward = chosen_policy_lp - chosen_ref_lp
    rejected_reward = rejected_policy_lp - rejected_ref_lp

    # Margin (standard DPO: chosen - rejected)
    margin = chosen_reward - rejected_reward

    # DPO loss
    loss = -stable_log_sigmoid(beta * margin)

    return {
        "loss": loss,
        "margin": margin,
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "chosen_policy_lp": chosen_policy_lp,
        "chosen_ref_lp": chosen_ref_lp,
        "rejected_policy_lp": rejected_policy_lp,
        "rejected_ref_lp": rejected_ref_lp,
        "beta": beta
    }


def compute_grpo_signals(traces: List[Dict], rankings: Dict[str, str]):
    """Compute GRPO optimization signals."""
    rank_rewards = {"Best": 1.0, "Middle": 0.5, "Worst": 0.0}

    rewards = {}
    for trace in traces:
        tid = trace["id"]
        rank = rankings.get(tid)
        rewards[tid] = rank_rewards.get(rank, 0.0)

    # Group statistics
    reward_vals = list(rewards.values())
    mean_reward = sum(reward_vals) / len(reward_vals)
    variance = sum((r - mean_reward) ** 2 for r in reward_vals) / len(reward_vals)
    std_reward = math.sqrt(variance)

    # Advantages
    advantages = {}
    for tid, reward in rewards.items():
        advantages[tid] = (reward - mean_reward) / (std_reward + 1e-8)

    return {
        "rewards": rewards,
        "advantages": advantages,
        "mean_reward": mean_reward,
        "std_reward": std_reward
    }


# ============== Main App ==============

def main():
    st.title("🏥 RLHF Workbench")
    st.markdown("""
    **Medical AI Post-Training Control Room**

    Rank AI reasoning traces and compute optimization signals using DPO and GRPO.
    """)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        st.subheader("DPO Settings")
        beta = st.slider("Beta (KL penalty)", 0.01, 1.0, 0.1, 0.01)

        st.subheader("GRPO Settings")
        st.caption("Rewards: Best=1.0, Middle=0.5, Worst=0.0")

    # Sample selector
    st.subheader("Select Medical Case")
    selected_id = st.selectbox(
        "Choose a sample:",
        list(SAMPLES.keys()),
        format_func=lambda x: f"{x} (Answer: {SAMPLES[x]['answer']})"
    )

    sample = SAMPLES[selected_id]
    traces = TRACES[selected_id]

    # Display question
    st.subheader("Medical Question")
    with st.expander("View Question", expanded=True):
        st.markdown(sample["question"])
        st.success(f"**Ground Truth: {sample['answer']}**")

    # Rank traces
    st.subheader("Rank the Reasoning Traces")
    st.markdown("Assign **Best**, **Middle**, and **Worst** to each trace.")

    # Session state for rankings
    if "rankings" not in st.session_state:
        st.session_state.rankings = {}
    if selected_id not in st.session_state.rankings:
        st.session_state.rankings[selected_id] = {}

    cols = st.columns(3)
    for i, (col, trace) in enumerate(zip(cols, traces)):
        with col:
            st.markdown(f"**Trace {i+1}**")
            with st.container(border=True, height=300):
                st.markdown(trace["text"][:800] + "..." if len(trace["text"]) > 800 else trace["text"])

            rank = st.selectbox(
                f"Rank",
                ["(Select)", "Best", "Middle", "Worst"],
                key=f"rank_{selected_id}_{trace['id']}"
            )
            if rank != "(Select)":
                st.session_state.rankings[selected_id][trace["id"]] = rank

    # Validate rankings
    rankings = st.session_state.rankings[selected_id]
    assigned = [r for r in rankings.values() if r]
    valid = len(assigned) == 3 and set(assigned) == {"Best", "Middle", "Worst"}

    if assigned and not valid:
        st.warning("Assign exactly one trace to each rank: Best, Middle, Worst")

    # Update button
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        clicked = st.button(
            "🔄 Compute Optimization Signals",
            type="primary",
            disabled=not valid,
            use_container_width=True
        )

    if clicked and valid:
        # Find chosen/rejected
        chosen_trace = next(t for t in traces if rankings[t["id"]] == "Best")
        rejected_trace = next(t for t in traces if rankings[t["id"]] == "Worst")

        # Compute signals
        dpo = compute_dpo_signals(chosen_trace["text"], rejected_trace["text"], beta)
        grpo = compute_grpo_signals(traces, rankings)

        st.success("Optimization signals computed!")

        # Display results
        tab1, tab2 = st.tabs(["DPO Results", "GRPO Results"])

        with tab1:
            st.subheader("Direct Preference Optimization")

            c1, c2 = st.columns(2)
            c1.metric("DPO Loss", f"{dpo['loss']:.4f}")
            c2.metric("Margin (chosen - rejected)", f"{dpo['margin']:.4f}")

            st.markdown("---")
            st.markdown("**Chosen (Best) Trace**")
            st.markdown(f"- Policy log-prob: `{dpo['chosen_policy_lp']:.2f}`")
            st.markdown(f"- Reference log-prob: `{dpo['chosen_ref_lp']:.2f}`")
            st.markdown(f"- **Implicit Reward**: `{dpo['chosen_reward']:.4f}`")

            st.markdown("**Rejected (Worst) Trace**")
            st.markdown(f"- Policy log-prob: `{dpo['rejected_policy_lp']:.2f}`")
            st.markdown(f"- Reference log-prob: `{dpo['rejected_ref_lp']:.2f}`")
            st.markdown(f"- **Implicit Reward**: `{dpo['rejected_reward']:.4f}`")

            st.info("**Interpretation**: Positive margin = model correctly prefers chosen trace (low loss).")

        with tab2:
            st.subheader("Group Relative Policy Optimization")

            c1, c2 = st.columns(2)
            c1.metric("Mean Reward", f"{grpo['mean_reward']:.4f}")
            c2.metric("Std Reward", f"{grpo['std_reward']:.4f}")

            st.markdown("---")
            st.markdown("**Per-Trace Results**")

            for trace in traces:
                tid = trace["id"]
                rank = rankings[tid]
                reward = grpo["rewards"][tid]
                adv = grpo["advantages"][tid]

                with st.container(border=True):
                    st.markdown(f"**{tid}** ({rank})")
                    c1, c2 = st.columns(2)
                    c1.metric("Reward", f"{reward:.2f}")
                    c2.metric("Advantage", f"{adv:.4f}")


if __name__ == "__main__":
    main()
