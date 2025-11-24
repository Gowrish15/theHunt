import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from stable_baselines3 import PPO
from predator_prey_multi_env import PredatorPreyMultiEnv

MODEL_PATH = "multi_prey_predator.zip"


def difficulty_schedule(cycle: int, total_cycles: int) -> float:
    # training curriculum (same as before)
    max_diff = 0.3
    if total_cycles <= 1:
        return 0.0
    frac = (cycle - 1) / (total_cycles - 1)
    return float(min(max_diff, max_diff * frac))


def evaluate_model(model: PPO, episodes: int = 15, eval_diff: float = 0.2) -> float:
    """
    Run several episodes (no rendering) at a fixed difficulty and
    return average prey caught. This is our "true" performance metric.
    """
    env_eval = PredatorPreyMultiEnv(grid_size=30, n_prey=100, max_steps=400)
    env_eval.set_difficulty(eval_diff)

    total = 0.0
    for _ in range(episodes):
        obs, _ = env_eval.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env_eval.step(action)
        total += info["total_caught"]

    env_eval.close()
    return total / episodes


def load_or_create_model(env: PredatorPreyMultiEnv, reset: bool) -> PPO:
    if reset and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print(f"Deleted existing model: {MODEL_PATH}")

    if os.path.exists(MODEL_PATH) and not reset:
        print(f"Loading existing model from {MODEL_PATH}")
        return PPO.load(MODEL_PATH, env=env)

    print("Creating new PPO model")
    return PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=1024,
        batch_size=1024,
        ent_coef=0.01,
    )


def run_episode(env, model, cycle, difficulty, live=True, record=True):
    """
    10s live rollout for visualization + GIF.
    """
    obs, _ = env.reset()
    frames = []

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"Cycle {cycle} — Difficulty {difficulty:.2f}")

    predator = ax.scatter([], [], s=120, c="red", marker="s")
    prey = ax.scatter([], [], s=12, c="blue", marker="o")

    status = ax.text(
        0.01, 0.98, "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
    )

    if live:
        plt.ion()
        plt.show(block=False)

    start = time.time()
    while time.time() - start < 10:  # EXACT 10 seconds
        px, py = env.pred
        alive_positions = env.prey[env.prey_alive]

        predator.set_offsets([[px, py]])
        prey.set_offsets(alive_positions)

        status.set_text(
            f"Step: {env.steps} | Alive: {np.sum(env.prey_alive)} | Caught: {env.caught_total}"
        )

        fig.canvas.draw()
        if record:
            buf = np.asarray(fig.canvas.buffer_rgba())
            frames.append(buf[:, :, :3].copy())

        if live:
            plt.pause(0.03)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)

    if live:
        plt.ioff()
        plt.close(fig)

    return frames, env.caught_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete saved weights and start learning from scratch",
    )
    args = parser.parse_args()

    env = PredatorPreyMultiEnv(grid_size=30, n_prey=100, max_steps=100000)
    model = load_or_create_model(env, reset=args.reset)

    # baseline eval BEFORE any new training
    best_avg = evaluate_model(model, episodes=15, eval_diff=0.2)
    print(f"Initial avg caught over 15 eval episodes = {best_avg:.2f}")

    total_cycles = 10
    all_frames = []
    gif_counts = []
    eval_scores = []

    for cycle in range(1, total_cycles + 1):
        diff = difficulty_schedule(cycle, total_cycles)
        env.set_difficulty(diff)

        timesteps = 80_000
        print(f"\n=== Cycle {cycle}/{total_cycles} — diff={diff:.2f} ===")
        print(f"Training timesteps this cycle: {timesteps}")
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

        # 10s live episode (for GIF)
        frames, caught = run_episode(env, model, cycle, diff)
        all_frames.extend(frames)
        gif_counts.append(caught)
        print(f"Cycle {cycle}: GIF episode caught {caught} prey")

        # proper evaluation (no rendering, many episodes)
        avg_eval = evaluate_model(model, episodes=15, eval_diff=0.2)
        eval_scores.append(avg_eval)
        print(f"Cycle {cycle}: avg caught over 15 eval episodes = {avg_eval:.2f}")

        # keep only best-performing weights (monotonic eval)
        if avg_eval >= best_avg:
            print(" -> New best model, saving.")
            best_avg = avg_eval
            model.save(MODEL_PATH)
        else:
            print(" -> Worse than best, reloading previous best model.")
            model = PPO.load(MODEL_PATH, env=env)

    # final save (best is already on disk, but this keeps behaviour same)
    model.save(MODEL_PATH)
    print(f"\nFinal best avg eval score = {best_avg:.2f}")
    print(f"Saved best model to {MODEL_PATH}")

    if all_frames:
        imageio.mimsave("multiprey_training.gif", all_frames, fps=8)
        print("Saved multiprey_training.gif")

    print("\n=== Summary (GIF episodes, noisy) ===")
    for i, c in enumerate(gif_counts, start=1):
        print(f"Cycle {i}: GIF caught {c} prey")

    print("\n=== Summary (Eval avg, monotonic by design) ===")
    for i, s in enumerate(eval_scores, start=1):
        print(f"Cycle {i}: eval avg = {s:.2f}")


if __name__ == "__main__":
    main()
