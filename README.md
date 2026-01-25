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

## Technical Details

### DPO Implementation

The DPO loss is computed as:

```
loss = -log(sigmoid(beta * margin))
```

Where:
- `margin = chosen_reward - rejected_reward` (standard DPO formulation)
- `reward = policy_log_prob - reference_log_prob`
- Reference model uses **pretrained weights** (same as initial policy, standard DPO)

The loss is low when the model correctly prefers the chosen trace, and high otherwise.

### GRPO Implementation

Group advantages are computed as:

```
advantage_i = (reward_i - mean_reward) / (std_reward + eps)
```

With optional healthcare safety shaping that boosts traces containing clinical escalation language (e.g., "consult", "emergency", "urgent").

### Log-Probability Computation

Log-probs are computed token-by-token over the completion only, conditioned on the full prompt and previous tokens. Both sum and length-normalized modes are supported, with caching to avoid recomputation.

### Distinctness Enforcement

Traces must be sufficiently distinct beyond trivial string differences:
- Normalized text hash collision detection
- Token-level Jaccard distance (minimum 0.3)
- Optional embedding-based semantic separation

## Data Format

Each sample JSON file should contain:

```json
{
  "Questions": "Full question text with multiple choice options...",
  "Answer": "B",
  "Reasoning": "Optional example reasoning trace..."
}
```

The system handles schema variations (e.g., `question` vs `Questions`) and fails loudly with clear errors if required fields are missing.

## Development

### Running Tests

```bash
# Test the optimizer directly
python optimizer.py

# Test the sampler
python sampler.py
```

### Adding New Model Backends

1. Create a new class implementing `ModelInterface`
2. Implement `generate()` and `compute_log_prob()` methods
3. Add backend enum value to `config.py`
4. Update `create_model()` factory function

## Troubleshooting

**"No pre-generated traces found"**: Run `python sampler.py` first to generate traces.

**Memory issues**: Use a smaller model (e.g., `distilgpt2`) or switch to API backend.

**Slow generation**: Use the trace cache to avoid regeneration. For APIs with rate limits, set `post_generation_sleep_seconds` in config.

## Architecture Notes

- **Model-Agnostic**: All LLM interactions go through `ModelInterface` abstraction
- **Caching**: Log-prob computations are cached using deterministic hashes
- **Resumable**: Trace generation can be interrupted and resumed from cache
- **Healthcare-Aware**: Optional safety shaping rewards clinically cautious reasoning
