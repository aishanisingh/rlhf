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

## Installation

### Local Setup

```bash
# Clone or navigate to the project directory
cd TriFetch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab

```python
!pip install torch transformers streamlit openai

# Upload the project files or clone from repository
# Then run the sampler and app
```

## Usage

### Step 1: Generate Reasoning Traces

Before using the UI, generate verified reasoning traces for each sample:

```bash
python sampler.py
```

This will:
- Load medical QA samples from `sample[1-5].json`
- Generate candidate reasoning traces using the model
- Perform rejection sampling to keep only correct answers
- Ensure traces are distinct (Jaccard distance > 0.3)
- Cache results to `.trace_cache/` for resumption

**Note**: The sampler supports optional rate limiting via `post_generation_sleep_seconds` config.

### Step 2: Launch the Workbench UI

```bash
streamlit run app.py
```

This opens a web interface where you can:
- Select a medical QA sample
- View the question and ground truth answer
- See three AI-generated reasoning traces
- Rank the traces as Best, Middle, and Worst
- Click "Update Model" to compute optimization signals

### Step 3: Review Optimization Signals

After ranking and clicking "Update Model", the UI displays:

**DPO Results:**
- DPO Loss value
- Margin (rejected_reward - chosen_reward)
- Implicit rewards for chosen and rejected traces
- Policy and reference log-probabilities

**GRPO Results:**
- Per-trace advantages
- Group mean and standard deviation
- Optional safety bonuses for traces with escalation language

## Configuration

Configuration can be modified in `config.py` or via environment variables with `TRIFETCH_` prefix.

### Key Settings

**DPO Configuration:**
- `beta`: KL penalty coefficient (default: 0.1)
- `use_length_scaling`: Enable token-length-aware beta scaling
- `log_prob_mode`: `sum` or `length_normalized`

**GRPO Configuration:**
- `reward_best/middle/worst`: Rank-based rewards (default: 1.0, 0.5, 0.0)
- `use_exponential_decay`: Alternative exponential reward decay
- `use_safety_shaping`: Boost traces with healthcare escalation language

**Model Configuration:**
- `local_model_name`: HuggingFace model ID (default: `distilgpt2`)
- `backend`: `local_transformers`, `openai`, `vllm`, or `groq`

### Switching Model Backends

To use an OpenAI-compatible API:

```python
# In config.py or via environment
config.model.backend = ModelBackend.OPENAI
config.model.api_base_url = "https://api.openai.com/v1"
config.model.api_key = "your-api-key"
config.model.api_model_name = "gpt-3.5-turbo-instruct"
```

For vLLM serving:

```python
config.model.backend = ModelBackend.VLLM
config.model.api_base_url = "http://localhost:8000/v1"
```

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
