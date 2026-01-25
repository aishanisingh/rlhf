# TriFetch

An RLHF workbench for post-training medical AI models using human preference feedback. Computes optimization signals using Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO).

## What it does

A pipeline for medical QA preference learning:

1. Generate multiple reasoning traces for each question
2. Have humans rank them (best, middle, worst)
3. Compute optimization signals based on the rankings

The goal is to understand the math behind RLHF, not to train a model end-to-end.

## Files

- `config.py` - configuration and hyperparameters
- `model_interface.py` - abstraction layer for different LLM backends
- `sampler.py` - trace generation with rejection sampling
- `optimizer.py` - DPO and GRPO computation logic
- `streamlit_app.py` - web interface
- `sample[1-5].json` - medical QA samples

## Setup

```bash
cd TriFetch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Generate traces first:
```bash
python sampler.py
```

2. Run the UI:
```bash
streamlit run streamlit_app.py
```

3. Select a medical question, rank the 3 AI responses (best/middle/worst), and click compute

You'll see DPO loss + margin, and GRPO advantages for each trace.

## Configuration

Edit `config.py` to change settings:

- `beta` - KL penalty coefficient, controls how much the policy can diverge from reference (default 0.1)
- `reward_best/middle/worst` - reward values assigned to each rank (1.0, 0.5, 0.0)
- `local_model_name` - HuggingFace model to use for local inference
- `backend` - model backend: `local_transformers`, `openai`, `vllm`, or `groq`

## Optimization Math

### DPO (Direct Preference Optimization)

DPO learns from pairwise preferences without explicitly training a reward model. The key idea is that we can derive an implicit reward from the policy itself.

**Implicit reward:**
```
r(x, y) = log(pi_policy(y|x)) - log(pi_ref(y|x))
```

This measures how much more likely the policy makes a response compared to the reference model. If the policy strongly prefers a response that the reference doesn't, the reward is high.

**Loss function:**
```
loss = -log(sigmoid(beta * (r_chosen - r_rejected)))
```

- When `r_chosen > r_rejected`: sigmoid output is high, loss is low (good)
- When `r_chosen < r_rejected`: sigmoid output is low, loss is high (needs training)
- `beta` controls sensitivity - higher beta means sharper preference distinctions

**Margin:** The difference `r_chosen - r_rejected` tells you how much the model already prefers the better response. Positive margin = model agrees with human ranking.

### GRPO (Group Relative Policy Optimization)

GRPO works with groups of responses rather than pairs. Instead of just comparing best vs worst, it considers all responses together.

**Reward assignment:**
Each response gets a reward based on its rank (best=1.0, middle=0.5, worst=0.0).

**Advantage computation:**
```
advantage_i = (reward_i - mean_reward) / (std_reward + eps)
```

This normalizes rewards relative to the group:
- Positive advantage = better than average
- Negative advantage = worse than average
- The std normalization ensures consistent gradient magnitudes across different reward distributions

GRPO is useful when you have more than 2 responses to compare, since it uses information from all of them rather than just the extremes.

## Data Format

Sample JSON files should have this structure:
```json
{
  "Questions": "Full medical question with multiple choice options...",
  "Answer": "B",
  "Reasoning": "Optional example reasoning..."
}
```

## Troubleshooting

- **"No traces found"** - Run `python sampler.py` first to generate traces
- **Out of memory** - Use a smaller model (e.g., `distilgpt2`) or switch to an API backend
