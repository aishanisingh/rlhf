"""
TriFetch RLHF Workbench - Simple Local UI
"""
import gradio as gr
import json
import os
import math

# ============== Data ==============

SAMPLES = {
    "sample1": {
        "question": """Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?

A) Mitral Valve Prolapse
B) Patent Foramen Ovale
C) Hypertrophic Cardiomyopathy
D) Ventricular Septal Defect""",
        "answer": "B",
        "topic": "Cardiology - Paradoxical Embolism"
    },
    "sample2": {
        "question": """A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. Her pulse is 110/min, respirations 22/min, and blood pressure 90/65 mm Hg. There is a 5-cm deep stab wound at the upper border of the 8th rib in the left midaxillary line. Which anatomical structure is most likely to be injured?

A) Left atrium of the heart
B) Lower lobe of the left lung
C) Spleen
D) Left lobe of the liver""",
        "answer": "B",
        "topic": "Anatomy - Thoracic Trauma"
    },
    "sample3": {
        "question": """A patient presents with progressive gait disturbances, tremors, and speech difficulties. Genetic testing confirms the presence of GAA trinucleotide repeat expansions. Which chromosome is most commonly associated with the mutated gene in this condition?

A) Chromosome 4
B) Chromosome 6
C) Chromosome 9
D) Chromosome X""",
        "answer": "C",
        "topic": "Genetics - Friedreich's Ataxia"
    },
}

TRACES = {
    "sample1": [
        {"id": "trace_0", "text": """Let me analyze this case systematically. The patient presents with sudden weakness in the left arm and leg, which suggests a neurological event - likely a stroke affecting the right side of the brain.

Key observation: The patient has a swollen, tender right lower leg after long-distance travel. This is classic for deep vein thrombosis (DVT).

Now, the critical question is: how could a venous clot cause an arterial stroke? Normally, venous blood goes through the lungs where clots would be filtered. However, if there's a right-to-left cardiac shunt, a clot could bypass the lungs entirely.

A Patent Foramen Ovale (PFO) is exactly such a shunt - a hole between the right and left atria that persists from fetal circulation. It's present in about 25% of adults.

The answer is B."""},
        {"id": "trace_1", "text": """This patient has neurological symptoms (left-sided weakness) combined with signs of DVT (swollen leg after travel).

When we see stroke symptoms in a young patient with DVT, we should think about paradoxical embolism. This happens when a venous clot crosses to the arterial circulation through a cardiac defect.

Looking at the options:
- Mitral Valve Prolapse: causes regurgitation, not shunting
- Patent Foramen Ovale: allows right-to-left shunting - this fits!
- Hypertrophic Cardiomyopathy: causes outflow obstruction
- VSD: usually left-to-right shunt in adults

The answer is B."""},
        {"id": "trace_2", "text": """The patient has weakness on one side and a swollen leg. The leg swelling after travel suggests a clot.

I need to find a heart problem that connects these. A hole in the heart could let a clot go to the brain.

Patent Foramen Ovale is a hole between atria.

The answer is B."""}
    ],
    "sample2": [
        {"id": "trace_0", "text": """This is a trauma case requiring careful anatomical analysis. Let me work through this systematically.

The wound is at the upper border of the 8th rib in the left midaxillary line, with 5cm depth. I need to consider what structures lie in this region.

Anatomical considerations:
- The 8th rib at the midaxillary line is at approximately the level of T8-T9
- This is below the cardiac silhouette but still within the thoracic cavity
- The lower lobe of the left lung extends to about the 10th rib in the midaxillary line

Given the location and depth, the lower lobe of the left lung is directly in the path.

The answer is B."""},
        {"id": "trace_1", "text": """A stab wound at the 8th rib, left midaxillary line, 5cm deep. The patient is hypotensive and tachycardic.

At this level on the left side, the main structures are:
- Lower lobe of left lung
- Possibly the diaphragm
- The spleen is nearby but more posterior

The clinical picture with unstable vitals suggests bleeding into the chest, most likely from the lung.

The answer is B."""},
        {"id": "trace_2", "text": """Stab wound on the left side of the chest at rib 8. The patient's blood pressure is low.

At this location, the lung would be hit. The lower lobe of the left lung is in this area.

The answer is B."""}
    ],
    "sample3": [
        {"id": "trace_0", "text": """The clinical presentation of progressive gait disturbances, tremors, and speech difficulties points toward a neurodegenerative condition affecting the cerebellum and its connections.

The genetic finding of GAA trinucleotide repeat expansions is the definitive clue. This pattern is pathognomonic for Friedreich's ataxia.

Let me recall the genetics:
- Friedreich's ataxia is caused by mutations in the FXN gene
- The FXN gene encodes frataxin, a mitochondrial protein
- The gene is located on chromosome 9q21

The answer is C."""},
        {"id": "trace_1", "text": """GAA trinucleotide repeats are characteristic of Friedreich's ataxia. This autosomal recessive condition affects the nervous system and heart.

The responsible gene is FXN (frataxin), and I need to remember which chromosome it's on.

Friedreich's ataxia is associated with chromosome 9 - specifically 9q21 where the FXN gene is located.

The answer is C."""},
        {"id": "trace_2", "text": """GAA repeats cause Friedreich's ataxia. The gene for this is on chromosome 9.

The answer is C."""}
    ],
}

# ============== Optimization Logic ==============

def stable_log_sigmoid(x: float) -> float:
    if x >= 0:
        return -math.log(1 + math.exp(-x) + 1e-8)
    else:
        return x - math.log(1 + math.exp(x) + 1e-8)


def compute_dpo(chosen_text: str, rejected_text: str, beta: float = 0.1):
    def sim_logprob(text: str, is_policy: bool) -> float:
        base = -2.0 * len(text.split())
        if is_policy:
            base += len(text.split()) * 0.1
        return base

    chosen_policy = sim_logprob(chosen_text, True)
    chosen_ref = sim_logprob(chosen_text, False)
    rejected_policy = sim_logprob(rejected_text, True)
    rejected_ref = sim_logprob(rejected_text, False)

    chosen_reward = chosen_policy - chosen_ref
    rejected_reward = rejected_policy - rejected_ref
    margin = chosen_reward - rejected_reward
    loss = -stable_log_sigmoid(beta * margin)

    return loss, margin, chosen_reward, rejected_reward


def compute_grpo(rewards_dict):
    vals = list(rewards_dict.values())
    mean_r = sum(vals) / len(vals)
    var = sum((r - mean_r) ** 2 for r in vals) / len(vals)
    std_r = math.sqrt(var)

    advantages = {}
    for k, r in rewards_dict.items():
        advantages[k] = (r - mean_r) / (std_r + 1e-8)

    return mean_r, std_r, advantages


# ============== UI Logic ==============

def get_sample_choices():
    return [f"{k}: {v['topic']}" for k, v in SAMPLES.items()]


def load_sample(choice):
    sample_id = choice.split(":")[0]
    sample = SAMPLES[sample_id]
    traces = TRACES[sample_id]

    return (
        sample["question"],
        sample["answer"],
        traces[0]["text"],
        traces[1]["text"],
        traces[2]["text"],
        sample_id
    )


def compute_signals(sample_id, rank1, rank2, rank3):
    if not sample_id:
        return "select a sample first", "", ""

    ranks = [rank1, rank2, rank3]
    if sorted(ranks) != ["best", "middle", "worst"]:
        return "assign each rank exactly once (best, middle, worst)", "", ""

    traces = TRACES[sample_id]

    # find chosen and rejected
    chosen_idx = ranks.index("best")
    rejected_idx = ranks.index("worst")

    chosen_text = traces[chosen_idx]["text"]
    rejected_text = traces[rejected_idx]["text"]

    # compute dpo
    loss, margin, chosen_r, rejected_r = compute_dpo(chosen_text, rejected_text)

    dpo_result = f"""loss: {loss:.4f}
margin: {margin:+.4f}

chosen reward: {chosen_r:+.4f}
rejected reward: {rejected_r:+.4f}

{"model agrees with your ranking" if margin > 0 else "model disagrees - training needed"}"""

    # compute grpo
    rank_to_reward = {"best": 1.0, "middle": 0.5, "worst": 0.0}
    rewards = {f"response {i+1}": rank_to_reward[r] for i, r in enumerate(ranks)}

    mean_r, std_r, advantages = compute_grpo(rewards)

    grpo_result = f"""mean reward: {mean_r:.2f}
std reward: {std_r:.2f}

advantages:"""

    for i, r in enumerate(ranks):
        adv = advantages[f"response {i+1}"]
        grpo_result += f"\n  response {i+1} ({r}): {adv:+.4f}"

    return "computed!", dpo_result, grpo_result


# ============== Build UI ==============

with gr.Blocks(title="TriFetch RLHF Workbench", theme=gr.themes.Soft()) as app:
    gr.Markdown("# TriFetch RLHF Workbench")
    gr.Markdown("rank ai responses to compute optimization signals")

    sample_id = gr.State("")

    with gr.Row():
        sample_dropdown = gr.Dropdown(
            choices=get_sample_choices(),
            label="select a medical case",
            scale=2
        )
        load_btn = gr.Button("load", scale=1)

    with gr.Row():
        with gr.Column():
            question_box = gr.Textbox(label="question", lines=6, interactive=False)
            answer_box = gr.Textbox(label="correct answer", interactive=False)

    gr.Markdown("---")
    gr.Markdown("### rank the responses (best, middle, worst)")

    with gr.Row():
        with gr.Column():
            trace1 = gr.Textbox(label="response 1", lines=8, interactive=False)
            rank1 = gr.Radio(["best", "middle", "worst"], label="rank")
        with gr.Column():
            trace2 = gr.Textbox(label="response 2", lines=8, interactive=False)
            rank2 = gr.Radio(["best", "middle", "worst"], label="rank")
        with gr.Column():
            trace3 = gr.Textbox(label="response 3", lines=8, interactive=False)
            rank3 = gr.Radio(["best", "middle", "worst"], label="rank")

    compute_btn = gr.Button("compute optimization signals", variant="primary")

    gr.Markdown("---")
    gr.Markdown("### results")

    with gr.Row():
        status = gr.Textbox(label="status", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("**DPO** (direct preference optimization)")
            dpo_output = gr.Textbox(label="", lines=8, interactive=False)
        with gr.Column():
            gr.Markdown("**GRPO** (group relative policy optimization)")
            grpo_output = gr.Textbox(label="", lines=8, interactive=False)

    # wire up events
    load_btn.click(
        load_sample,
        inputs=[sample_dropdown],
        outputs=[question_box, answer_box, trace1, trace2, trace3, sample_id]
    )

    compute_btn.click(
        compute_signals,
        inputs=[sample_id, rank1, rank2, rank3],
        outputs=[status, dpo_output, grpo_output]
    )


if __name__ == "__main__":
    app.launch()
