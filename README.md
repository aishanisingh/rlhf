# trifetch

a post-training workbench for medical ai, focused on human preference optimization using dpo and grpo.

## why this matters for healthcare

in medical ai, **the reasoning path matters as much as the answer**. two models can both output "B" for a diagnosis question, but one might reason through the differential systematically while the other guesses. post-training with human feedback lets us reward the better reasoning process, not just correctness.

this project demonstrates:
- **preference learning** over reasoning traces, not just final answers
- **rejection sampling** to ensure all candidate traces reach the correct answer (we only rank *how* they got there)
- **safety-aware reward shaping** that can boost traces containing clinical escalation language

## post-training approach

this implements two modern preference optimization methods that have largely replaced ppo-based rlhf:

**dpo (direct preference optimization):**
- skips the reward model entirely - derives implicit rewards from policy/reference log-prob ratios
- more stable training than ppo, fewer hyperparameters to tune
- the loss directly optimizes for the bradley-terry preference model

**grpo (group relative policy optimization):**
- extends beyond pairwise comparisons to handle groups of responses
- normalizes advantages relative to the group, reducing variance
- better utilizes data when you have multiple candidates per prompt

both methods fit naturally after sft (supervised fine-tuning) in the post-training pipeline: pretrain → sft → preference optimization.

## what it does

1. generate multiple reasoning traces for each medical question
2. humans rank them by reasoning quality (best, middle, worst)
3. compute optimization signals that would train the model to prefer better reasoning

the ui shows the computed loss, margin, and advantages - the actual gradients that would update the model.

## files

- `app.py` - streamlit web ui for interactive ranking
- `main.py` - cli with real model log-probabilities
- `optimizer.py` - dpo and grpo math (the core logic)
- `model_interface.py` - abstraction for local/api model backends
- `sampler.py` - trace generation with rejection sampling
- `config.py` - hyperparameters (beta, rewards, model settings)
- `sample[1-5].json` - medical qa cases

## setup

```bash
cd TriFetch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## usage

### web ui

```bash
streamlit run app.py
```

or if streamlit isn't in your path:

```bash
python -m streamlit run app.py
```

opens at `http://localhost:8501`. select a case, rank the 3 traces, compute signals.

### cli with real models

```bash
python main.py
```

uses actual model log-probs instead of simulated values. slower but gives real optimization signals.

## the math

### dpo loss

```
loss = -log(sigmoid(beta * margin))
margin = r_chosen - r_rejected
r(x,y) = log(pi_policy(y|x)) - log(pi_ref(y|x))
```

the implicit reward `r(x,y)` measures how much more the policy likes a response compared to the reference. positive margin means the model already prefers the better trace.

### grpo advantages

```
advantage_i = (reward_i - mean_reward) / (std_reward + eps)
```

normalizes rewards relative to the group. best trace gets positive advantage, worst gets negative. the std normalization keeps gradient magnitudes consistent.

## healthcare considerations

**why reasoning traces matter:**
- a model that gets the right answer for wrong reasons will fail on edge cases
- in healthcare, we need to trust the reasoning path, not just the output
- ranking traces by reasoning quality trains models to think systematically

**rejection sampling:**
- we only show traces that reach the correct answer
- this isolates reasoning quality from answer correctness
- humans rank *how* the model reasons, not *what* it concludes

**safety shaping (optional):**
- config supports boosting traces with escalation language ("consult specialist", "emergency", "urgent")
- rewards clinically cautious reasoning in ambiguous cases

## configuration

key settings in `config.py`:

- `beta` - kl penalty (0.1 default). higher = sharper preference learning
- `reward_best/middle/worst` - rank rewards (1.0, 0.5, 0.0)
- `use_safety_shaping` - boost traces with clinical escalation language
- `local_model_name` - huggingface model for log-prob computation

## design choices

**why dpo over ppo?**
- simpler: no separate reward model, no value function, no clipping
- stable: direct optimization of the preference objective
- practical: fewer hyperparameters, easier to debug

**why show optimization signals instead of training?**
- demonstrates understanding of the math without gpu requirements
- focuses on the preference learning logic, not training infrastructure
- the signals shown are exactly what would update model weights

**why medical qa?**
- high stakes: reasoning quality directly impacts patient outcomes
- clear ranking criteria: systematic reasoning > pattern matching > guessing
- demonstrates healthcare-specific considerations in ai post-training
