import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env import RefinedPredatorPreyEnv
import os

# === CONFIGURATION ===
TOTAL_CYCLES = 25
STEPS_PER_CYCLE = 20_000 
N_ENVS = 8  # Parallel environments (Set to 4 or 8 depending on CPU cores)
MODEL_PATH = "evolved_predator_gpu.zip"
GRAPH_FILENAME = "coevolution_results_gpu.png"

def make_env(rank, difficulty=0.0):
    """Utility to create env with specific seed/difficulty"""
    def _init():
        env = RefinedPredatorPreyEnv(grid_size=20, n_prey=40, max_steps=300)
        env.set_difficulty(difficulty)
        return env
    return _init

def evaluate_model(model, env, episodes=5):
    """Eval runs on a single env instance for simplicity"""
    ratios = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        ratios.append(info.get("total_caught", 0) / env.n_prey)
    return np.mean(ratios)

def main():
    # 1. Hardware Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Create Vectorized Environment (Parallel CPU -> GPU Batch)
    # On Windows, SubprocVecEnv is heavier. If it crashes, switch to DummyVecEnv.
    vec_env = make_vec_env(
        RefinedPredatorPreyEnv, 
        n_envs=N_ENVS, 
        vec_env_cls=SubprocVecEnv 
    )

    # 3. Setup Model
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=vec_env, device=device)
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=0, 
            learning_rate=3e-4, 
            batch_size=2048,  # Larger batch size for GPU
            n_steps=2048 // N_ENVS,
            device=device
        )

    history = {"cycle": [], "difficulty": [], "catch_rate": []}
    current_difficulty = 0.1
    
    # Separate eval env (single instance)
    eval_env = RefinedPredatorPreyEnv(grid_size=20, n_prey=40)

    print(f"\nStarting GPU-Accelerated Training ({N_ENVS} parallel envs)...")
    print("-" * 65)
    print(f"{'Cycle':<6} | {'Difficulty':<12} | {'Catch Rate':<12} | {'Status'}")
    print("-" * 65)

    for cycle in range(1, TOTAL_CYCLES + 1):
        # Update difficulty across all parallel envs
        # We access the internal envs using env_method
        vec_env.env_method("set_difficulty", current_difficulty)
        eval_env.set_difficulty(current_difficulty)
        
        # Train (Fast!)
        model.learn(total_timesteps=STEPS_PER_CYCLE, reset_num_timesteps=False)
        
        # Evaluate (Single env for accuracy)
        avg_rate = evaluate_model(model, eval_env, episodes=5)
        
        history["cycle"].append(cycle)
        history["difficulty"].append(current_difficulty)
        history["catch_rate"].append(avg_rate)

        # Evolution Logic
        status = "---"
        if avg_rate > 0.75:
            current_difficulty += 0.1
            status = "Prey Evolved (Diff UP)"
        elif avg_rate < 0.25:
            current_difficulty -= 0.05
            status = "Prey Degraded (Diff DOWN)"
        
        current_difficulty = np.clip(current_difficulty, 0.0, 1.0)
        print(f"{cycle:<6} | {history['difficulty'][-1]:<12.2f} | {avg_rate:<12.2%} | {status}")
        
        if cycle % 5 == 0:
            model.save(MODEL_PATH)

    model.save(MODEL_PATH)
    vec_env.close()
    
    # Plotting
    generate_graph(history)

def generate_graph(history):
    # (Same plotting code as before)
    cycles = history["cycle"]
    diffs = history["difficulty"]
    rates = history["catch_rate"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Training Cycle')
    ax1.set_ylabel('Predator Catch Rate', color=color, fontweight='bold')
    ax1.plot(cycles, rates, color=color, marker='o', label="Predator Success")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Prey Difficulty', color=color, fontweight='bold')
    ax2.plot(cycles, diffs, color=color, linestyle='--', marker='x', label="Prey Difficulty")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)

    plt.title('GPU-Accelerated Co-Evolution')
    fig.tight_layout()
    plt.savefig(GRAPH_FILENAME)
    plt.show()

if __name__ == "__main__":
    main()