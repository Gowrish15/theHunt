import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from stable_baselines3 import PPO
from predator_prey_multi_env import PredatorPreyMultiEnv

MODEL_PATH = "multi_prey_predator.zip"


def main():
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Could not load model from {MODEL_PATH}: {e}")
        return

    env = PredatorPreyMultiEnv(grid_size=30, n_prey=100, max_steps=100000)
    env.set_difficulty(0.3)

    obs, _ = env.reset()
    frames = []

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    predator = ax.scatter([], [], s=120, c="red", marker="s")
    prey = ax.scatter([], [], s=12, c="blue", marker="o")

    status = ax.text(
        0.01, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        color="black",
    )

    for _ in range(400):
        px, py = env.pred
        alive_positions = env.prey[env.prey_alive]

        predator.set_offsets([[px, py]])
        prey.set_offsets(alive_positions)

        status.set_text(
            f"Step: {env.steps} | Alive: {np.sum(env.prey_alive)} | Caught: {env.caught_total}"
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[:, :, :3].copy())

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    imageio.mimsave("multiprey_replay.gif", frames, fps=8)
    print("Saved multiprey_replay.gif")


if __name__ == "__main__":
    main()
