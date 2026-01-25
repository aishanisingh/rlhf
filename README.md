# trifetch

an rlhf workbench for post-training medical ai models using human preference feedback. computes optimization signals using direct preference optimization (dpo) and group relative policy optimization (grpo).

## what it does

a pipeline for medical qa preference learning:

1. generate multiple reasoning traces for each question
2. have humans rank them (best, middle, worst)
3. compute optimization signals based on the rankings

the goal is to understand the math behind rlhf, not to train a model end-to-end.

## files

- `app.py` - local web ui (gradio)
- `main.py` - cli interface with real model log-probs
- `config.py` - configuration and hyperparameters
- `model_interface.py` - abstraction layer for different llm backends
- `sampler.py` - trace generation with rejection sampling
- `optimizer.py` - dpo and grpo computation logic
- `sample[1-5].json` - medical qa samples

## setup

```bash
cd TriFetch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## usage

### web ui (recommended)

```bash
python app.py
```

opens a local web interface at `http://127.0.0.1:7860` where you can:
- select from 5 medical cases
- view 3 ai-generated reasoning traces
- rank each trace (best, middle, worst) - all three must be ranked with unique values
- compute dpo and grpo optimization signals

### cli with real models

```bash
python main.py
```

uses actual model log-probabilities instead of simulated values. requires more memory and time but gives real optimization signals.

## configuration

edit `config.py` to change settings:

- `local_model_name` - huggingface model to use (default: `distilgpt2`)
- `beta` - kl penalty coefficient, controls how much the policy can diverge from reference (default 0.1)
- `reward_best/middle/worst` - reward values assigned to each rank (1.0, 0.5, 0.0)
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

- **"no samples with cached traces found"** (cli only) - run `python sampler.py` first to generate traces
- **out of memory** - use a smaller model in config.py (e.g., `distilgpt2`)
- **slow log-prob computation** - results are cached in `.logprob_cache_*.json` files, subsequent runs will be faster
