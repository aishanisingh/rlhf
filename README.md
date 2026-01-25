# TriFetch

an rlhf workbench for post-training medical AI with human feedback. uses DPO and GRPO to figure out which AI reasoning traces are better.

## what it does

basically a pipeline for medical QA:

1. generate a bunch of reasoning traces
2. have humans rank them (best, middle, worst)
3. compute optimization signals based on the rankings

the goal is to understand the math behind rlhf, not to actually train a model end-to-end.

## files

- `config.py` - all the settings
- `model_interface.py` - talks to different LLM backends
- `sampler.py` - generates reasoning traces
- `optimizer.py` - DPO and GRPO math
- `streamlit_app.py` - the web ui
- `sample[1-5].json` - medical questions

## setup

```bash
cd TriFetch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## how to use

1. generate traces first:
```bash
python sampler.py
```

2. run the ui:
```bash
streamlit run streamlit_app.py
```

3. pick a medical question, rank the 3 AI responses (best/middle/worst), hit compute

you'll see DPO loss + margin, and GRPO advantages for each trace

## config

edit `config.py` to change stuff:

- `beta` - how much to penalize diverging from reference (default 0.1)
- `reward_best/middle/worst` - rewards for each rank (1.0, 0.5, 0.0)
- `local_model_name` - which huggingface model to use
- `backend` - can be `local_transformers`, `openai`, `vllm`, or `groq`

## the math

**DPO**: `loss = -log(sigmoid(beta * margin))` where margin is how much the model prefers the chosen trace over rejected

**GRPO**: `advantage = (reward - mean) / std` - normalizes rewards relative to the group

## data format

sample json files look like:
```json
{
  "Questions": "the medical question...",
  "Answer": "B"
}
```

## if things break

- "no traces found" -> run `python sampler.py` first
- out of memory -> use a smaller model or switch to api backend
