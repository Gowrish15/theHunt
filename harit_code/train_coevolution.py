import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from predator_prey_multi_env import PredatorTrainingEnv, PreyTrainingEnv, SharedConfig

# === CONFIGURATION ===
N_CYCLES = 20         # Reduced from 50 to save time
N_ENVS = 8            # Parallel environments
OUTPUT_DIR = "output" # Directory to save GIFs

# Dynamic Training Config
PRED_TARGET_CATCH_PCT = 0.70
PREY_TARGET_SURVIVAL_PCT = 0.60
MAX_PHASE_STEPS = 100_000     # Reduced to 100k per cycle
CHECK_INTERVAL = 10_240       # Check less often to speed up training
N_EVAL_EPISODES = 10          # Standard eval

# Smaller Network for Faster Training
POLICY_KWARGS = dict(activation_fn=nn.ReLU, net_arch=[64, 64])

def generate_gif(pred_model, prey_model, cycle, filename=None):
    """Generates a GIF of a single episode between the two models."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if filename is None:
        filename = os.path.join(OUTPUT_DIR, f"cycle_{cycle}_replay.gif")
    
    print(f"Generating GIF: {filename}...")
    env = PredatorTrainingEnv(prey_policy=prey_model)
    obs, _ = env.reset()
    
    frames = []
    done = False
    step_count = 0
    
    # Setup Plot
    # Logic adapted from replay_to_gif.py (No Agg backend, no ion)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Initialize scatter plots (UI preserved: Red X, Blue dots)
    predator_scatter = ax.scatter([], [], c='red', s=100, marker='x', label='Predator')
    prey_scatter = ax.scatter([], [], c='blue', s=50, label='Prey')
    
    # Title text object
    title_text = ax.set_title(f"Cycle {cycle} | Step 0")
    
    # Draw initial state
    predator_scatter.set_offsets([env.pred])
    if np.any(env.prey_alive):
        prey_scatter.set_offsets(env.prey[env.prey_alive])
    else:
        prey_scatter.set_offsets(np.empty((0, 2)))
    
    while not done:
        step_count += 1
        
        # 1. Capture Frame (Draw current state)
        title_text.set_text(f"Cycle {cycle} | Step {step_count-1} | Caught: {env.caught_total}/{env.n_prey}")
        
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        image = buf[:, :, :3].copy() # RGB only
        frames.append(image)
        
        # 2. Logic (Step Environment)
        action, _ = pred_model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        
        # 3. Update Data for next draw
        predator_scatter.set_offsets([env.pred])
        
        if np.any(env.prey_alive):
            alive_prey = env.prey[env.prey_alive]
            prey_scatter.set_offsets(alive_prey)
        else:
            prey_scatter.set_offsets(np.empty((0, 2)))
            
    plt.close(fig)
    
    print(f"Captured {len(frames)} frames. Saving GIF...")
    if len(frames) > 0:
        imageio.mimsave(filename, frames, fps=10, loop=0)
        print(f"Saved {filename}")
    else:
        print("Error: No frames captured!")

def make_pred_env(prey_policy=None):
    def _init():
        return PredatorTrainingEnv(prey_policy)
    return _init

def make_prey_env(pred_policy=None, bg_prey_policy=None):
    def _init():
        return PreyTrainingEnv(pred_policy, bg_prey_policy)
    return _init

def evaluate_matchup(pred_model, prey_model, episodes=10):
    """
    Run a matchup between the two models to see who wins.
    Metric: Average Prey Caught per Episode.
    """
    # We use the PredatorTrainingEnv for evaluation as it tracks catches easily
    env = PredatorTrainingEnv(prey_policy=prey_model)
    
    total_caught = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = pred_model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        total_caught.append(info['total_caught'])
    
    return np.mean(total_caught)

def main():
    print("=== STARTING ADVERSARIAL CO-EVOLUTION ===")
    print(f"Predator Speed: 100% | Prey Speed: {100 * (1 - 1/SharedConfig.PREY_SPEED_MODIFIER):.1f}%")
    print(f"Grid Size: {SharedConfig.GRID_SIZE}x{SharedConfig.GRID_SIZE}")
    print(f"Prey Count: {SharedConfig.N_PREY}")
    print(f"Max Steps per Episode: {SharedConfig.MAX_STEPS}")
    
    # 1. Initialize Models
    # We need dummy envs to initialize the models initially
    # FIX: Create N_ENVS for the initial dummy envs so the model expects 8 envs
    dummy_pred_env = DummyVecEnv([make_pred_env(None) for _ in range(N_ENVS)])
    dummy_prey_env = DummyVecEnv([make_prey_env(None, None) for _ in range(N_ENVS)])
    
    print("Creating NEW Predator...")
    # n_steps is per env, so 256 * 8 = 2048 total buffer size
    pred_model = PPO("MlpPolicy", dummy_pred_env, verbose=0, batch_size=2048, n_steps=256, policy_kwargs=POLICY_KWARGS)

    print("Creating NEW Prey...")
    prey_model = PPO("MlpPolicy", dummy_prey_env, verbose=0, batch_size=2048, n_steps=256, policy_kwargs=POLICY_KWARGS)

    history = {"cycle": [], "avg_caught": []}

    # 2. Training Loop
    for cycle in range(1, N_CYCLES + 1):
        print(f"\n--- Cycle {cycle}/{N_CYCLES} ---")
        
        # --- PHASE 1: TRAIN PREDATOR ---
        print(f">> Training Predator (Goal: Catch >= {PRED_TARGET_CATCH_PCT:.0%})...")
        pred_train_env = DummyVecEnv([make_pred_env(prey_model) for _ in range(N_ENVS)])
        pred_model.set_env(pred_train_env)
        
        p_steps = 0
        while p_steps < MAX_PHASE_STEPS:
            pred_model.learn(total_timesteps=CHECK_INTERVAL, reset_num_timesteps=False)
            p_steps += CHECK_INTERVAL
            
            avg_caught = evaluate_matchup(pred_model, prey_model, episodes=N_EVAL_EPISODES)
            catch_rate = avg_caught / SharedConfig.N_PREY
            print(f"   [Predator] Steps: {p_steps} | Catch Rate: {catch_rate:.1%}")
            
            if catch_rate >= PRED_TARGET_CATCH_PCT:
                print("   -> Predator Goal Reached!")
                break
        pred_train_env.close()
        
        # --- PHASE 2: TRAIN PREY ---
        target_catch_rate = 1.0 - PREY_TARGET_SURVIVAL_PCT
        print(f">> Training Prey (Goal: Catch <= {target_catch_rate:.0%})...")
        prey_train_env = DummyVecEnv([make_prey_env(pred_model, prey_model) for _ in range(N_ENVS)])
        prey_model.set_env(prey_train_env)
        
        pr_steps = 0
        while pr_steps < MAX_PHASE_STEPS:
            prey_model.learn(total_timesteps=CHECK_INTERVAL, reset_num_timesteps=False)
            pr_steps += CHECK_INTERVAL
            
            avg_caught = evaluate_matchup(pred_model, prey_model, episodes=N_EVAL_EPISODES)
            catch_rate = avg_caught / SharedConfig.N_PREY
            print(f"   [Prey] Steps: {pr_steps} | Catch Rate: {catch_rate:.1%}")
            
            if catch_rate <= target_catch_rate:
                print("   -> Prey Goal Reached!")
                break
        prey_train_env.close()
        
        # --- EVALUATE ---
        score = evaluate_matchup(pred_model, prey_model)
        history["cycle"].append(cycle)
        history["avg_caught"].append(score)
        
        print(f"Result: Predator caught {score:.1f} / {SharedConfig.N_PREY} prey on average.")
        
        # Save periodically
        # if cycle % 5 == 0:
        #     pred_model.save(PRED_MODEL_PATH)
        #     prey_model.save(PREY_MODEL_PATH)
        #     print("Models saved.")
            
        # Generate GIF every cycle
        generate_gif(pred_model, prey_model, cycle)

    # Final Save
    # pred_model.save(PRED_MODEL_PATH)
    # prey_model.save(PREY_MODEL_PATH)
    print("Training Complete.")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history["cycle"], history["avg_caught"], marker='o')
    plt.title("Co-Evolution: Predator Catch Rate")
    plt.xlabel("Cycle")
    plt.ylabel("Avg Prey Caught (out of 50)")
    plt.grid(True)
    plt.savefig("coevolution_progress.png")
    print("Saved graph to coevolution_progress.png")

if __name__ == "__main__":
    main()
