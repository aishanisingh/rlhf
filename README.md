# trifetch

an rlhf workbench for post-training medical ai models using human preference feedback. computes optimization signals using direct preference optimization (dpo) and group relative policy optimization (grpo).

## what it does

a pipeline for medical qa preference learning:

1. generate multiple reasoning traces for each question
2. have humans rank them (best, middle, worst)
3. compute optimization signals based on the rankings

the goal is to understand the math behind rlhf, not to train a model end-to-end.

## files

- `config.py` - configuration and hyperparameters
- `model_interface.py` - abstraction layer for different llm backends
- `sampler.py` - trace generation with rejection sampling
- `optimizer.py` - dpo and grpo computation logic
- `streamlit_app.py` - web interface
- `sample[1-5].json` - medical qa samples

## setup

```bash
cd TriFetch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## usage

1. generate traces first:
```bash
python sampler.py
```

2. run the ui:
```bash
streamlit run streamlit_app.py
```

3. select a medical question, rank the 3 ai responses (best/middle/worst), and click compute

you'll see dpo loss + margin, and grpo advantages for each trace.

## configuration

edit `config.py` to change settings:

- `beta` - kl penalty coefficient, controls how much the policy can diverge from reference (default 0.1)
- `reward_best/middle/worst` - reward values assigned to each rank (1.0, 0.5, 0.0)
- `local_model_name` - huggingface model to use for local inference
- `backend` - model backend: `local_transformers`, `openai`, `vllm`, or `groq`

## optimization math

### dpo (direct preference optimization)

dpo learns from pairwise preferences without explicitly training a reward model. the key idea is that we can derive an implicit reward from the policy itself.

**implicit reward:**
```
r(x, y) = log(pi_policy(y|x)) - log(pi_ref(y|x))
```

this measures how much more likely the policy makes a response compared to the reference model. if the policy strongly prefers a response that the reference doesn't, the reward is high.

**loss function:**
```
loss = -log(sigmoid(beta * (r_chosen - r_rejected)))
```

- when `r_chosen > r_rejected`: sigmoid output is high, loss is low (good)
- when `r_chosen < r_rejected`: sigmoid output is low, loss is high (needs training)
- `beta` controls sensitivity - higher beta means sharper preference distinctions

**margin:** the difference `r_chosen - r_rejected` tells you how much the model already prefers the better response. positive margin = model agrees with human ranking.

### grpo (group relative policy optimization)

grpo works with groups of responses rather than pairs. instead of just comparing best vs worst, it considers all responses together.

**reward assignment:**
each response gets a reward based on its rank (best=1.0, middle=0.5, worst=0.0).

**advantage computation:**
```
advantage_i = (reward_i - mean_reward) / (std_reward + eps)
```

this normalizes rewards relative to the group:
- positive advantage = better than average
- negative advantage = worse than average
- the std normalization ensures consistent gradient magnitudes across different reward distributions

grpo is useful when you have more than 2 responses to compare, since it uses information from all of them rather than just the extremes.

## data format

sample json files should have this structure:
```json
{
  "Questions": "full medical question with multiple choice options...",
  "Answer": "B",
  "Reasoning": "optional example reasoning..."
}
```

## troubleshooting

- **"no traces found"** - run `python sampler.py` first to generate traces
- **out of memory** - use a smaller model (e.g., `distilgpt2`) or switch to an api backend
