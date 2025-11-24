
# The Hunt – Curriculum-Based Predator–Prey RL

This project implements a simple predator–prey gridworld in Gymnasium and trains
a predator agent using PPO (Stable-Baselines3) against a prey whose evasiveness
increases over time (curriculum learning).

## Files

- `predator_prey_env.py` – Custom Gymnasium environment with:
  - Predator as the RL agent.
  - Prey controlled by a policy whose evasiveness is controlled by `difficulty`.
- `train_predator.py` – Baseline training of the predator vs purely random prey.
- `live_curriculum_train.py` – Main script:
  - Runs curriculum training over multiple cycles.
  - In each cycle, prey difficulty increases.
  - After each cycle, one rollout is rendered.
  - All rollouts are combined into a single GIF (`curriculum_training.gif`).
- `evaluate_predator.py` – Evaluates the baseline predator model against
  weak and fully evasive prey.
- `requirements.txt` – Python dependencies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Baseline predator (optional)

Train a predator against random prey (difficulty = 0.0):

```bash
python train_predator.py
```

This saves `predator_vs_random.zip`.

### 2. Curriculum training with visualization (main)

Run the curriculum training and generate a GIF showing all cycles:

```bash
python live_curriculum_train.py
```

This will:

- Train PPO in multiple cycles.
- Gradually increase prey difficulty.
- Record one rollout after each cycle.
- Save the combined animation to `curriculum_training.gif`.
- Print a summary table with difficulty, catch indicator, and steps per cycle.

### 3. Evaluate predator (baseline model)

If you trained the baseline predator with `train_predator.py`, you can evaluate it:

```bash
python evaluate_predator.py
```

This prints success rate and average steps vs:
- Random prey (`difficulty=0.0`)
- Evasive prey (`difficulty=1.0`)

## Notes

- The visualization has no gridlines for a cleaner look.
- The environment is kept small (5x5) so training is fast on CPU.
- Curriculum training is stochastic; exact numbers will vary between runs.
