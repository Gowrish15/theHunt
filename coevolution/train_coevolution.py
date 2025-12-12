import os
import json
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from predator_prey_multi_env import PredatorTrainingEnv, PreyTrainingEnv, SharedConfig

# === CONFIGURATION ===
RUN_NAME = "toroidal_with_obstacles"  # Change this for different experiments
N_CYCLES = 20
N_ENVS = 12                # Match your 12 threads
OUTPUT_DIR = os.path.join("output", RUN_NAME)
MODEL_DIR = os.path.join("models", RUN_NAME)

# Dynamic Training Config
PRED_TARGET_CATCH_PCT = 0.70
PREY_TARGET_SURVIVAL_PCT = 0.60
MAX_PHASE_STEPS = 50_000   # Reduced from 100k
CHECK_INTERVAL = 25_000    # Doubled - check less often
N_EVAL_EPISODES = 5        # Reduced from 10

# Smaller Network for Faster Training
POLICY_KWARGS = dict(activation_fn=nn.ReLU, net_arch=[64, 64])

def generate_gif(pred_model, prey_model, cycle, filename=None, max_frames=1000):
    """Generates a GIF of a single episode between the two models."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if filename is None:
        filename = os.path.join(OUTPUT_DIR, f"cycle_{cycle}.gif")
    
    print(f"Generating GIF: {filename}...")
    env = PredatorTrainingEnv(prey_policy=prey_model)
    obs, _ = env.reset()
    
    frames = []
    done = False
    step_count = 0
    
    # Setup Plot with fixed DPI for consistent frame sizes
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.tight_layout(pad=0.5)
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Draw obstacles (static, only once)
    obs_y, obs_x = np.where(env.obstacles)
    if len(obs_x) > 0:
        ax.scatter(obs_x, obs_y, c='gray', s=30, marker='s', alpha=0.5)
    
    # Initialize scatter plots
    predator_scatter = ax.scatter([], [], c='red', s=100, marker='x', label='Predator')
    prey_scatter = ax.scatter([], [], c='blue', s=50, label='Prey')
    
    # Title text object
    title_text = ax.set_title(f"Cycle {cycle} | Step 0")
    
    # Draw initial state (positions are [row,col]=[y,x], but scatter wants [x,y])
    predator_scatter.set_offsets([[env.pred[1], env.pred[0]]])
    if np.any(env.prey_alive):
        alive_prey = env.prey[env.prey_alive]
        prey_scatter.set_offsets(alive_prey[:, ::-1])  # Swap columns to get [x,y]
    else:
        prey_scatter.set_offsets(np.empty((0, 2)))
    
    # Get expected frame size from first render
    fig.canvas.draw()
    first_buf = np.asarray(fig.canvas.buffer_rgba())
    expected_shape = first_buf.shape[:2]
    
    while not done and step_count < max_frames:
        step_count += 1
        
        # 1. Capture Frame
        title_text.set_text(f"Cycle {cycle} | Step {step_count-1} | Caught: {env.caught_total}/{env.n_prey}")
        
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        
        # Ensure consistent frame size
        if buf.shape[:2] == expected_shape:
            image = buf[:, :, :3].copy()
            frames.append(image)
        
        # 2. Step Environment
        action, _ = pred_model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        
        # 3. Update Data for next draw (swap [row,col] to [x,y])
        predator_scatter.set_offsets([[env.pred[1], env.pred[0]]])
        
        if np.any(env.prey_alive):
            alive_prey = env.prey[env.prey_alive]
            prey_scatter.set_offsets(alive_prey[:, ::-1])  # Swap columns
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

def evaluate_matchup(pred_model, prey_model, episodes=10, detailed=False):
    """
    Run a matchup between the two models.
    If detailed=True, returns full stats dict. Otherwise just avg caught.
    """
    env = PredatorTrainingEnv(prey_policy=prey_model)
    
    stats = {
        'catches': [],
        'steps': [],
        'first_catch_step': [],
        'predator_starved': 0,
        'all_prey_caught': 0,
        'truncated': 0,
    }
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        first_catch = None
        
        while not done:
            action, _ = pred_model.predict(obs, deterministic=True)
            old_caught = env.caught_total
            obs, _, term, trunc, info = env.step(action)
            
            if env.caught_total > old_caught and first_catch is None:
                first_catch = env.steps
            
            done = term or trunc
        
        stats['catches'].append(info['total_caught'])
        stats['steps'].append(env.steps)
        stats['first_catch_step'].append(first_catch if first_catch else env.steps)
        
        if env.pred_energy <= 0:
            stats['predator_starved'] += 1
        elif not np.any(env.prey_alive):
            stats['all_prey_caught'] += 1
        else:
            stats['truncated'] += 1
    
    if detailed:
        return {
            'avg_caught': np.mean(stats['catches']),
            'std_caught': np.std(stats['catches']),
            'avg_steps': np.mean(stats['steps']),
            'avg_first_catch': np.mean(stats['first_catch_step']),
            'predator_starved': stats['predator_starved'],
            'all_prey_caught': stats['all_prey_caught'],
            'truncated': stats['truncated'],
            'catch_rate': np.mean(stats['catches']) / SharedConfig.N_PREY,
        }
    return np.mean(stats['catches'])

def main():
    print("=== STARTING ADVERSARIAL CO-EVOLUTION ===")
    print(f"Predator Speed: 100% | Prey Speed: {100 * (1 - 1/SharedConfig.PREY_SPEED_MODIFIER):.1f}%")
    print(f"Grid Size: {SharedConfig.GRID_SIZE}x{SharedConfig.GRID_SIZE}")
    print(f"Prey Count: {SharedConfig.N_PREY}")
    print(f"Max Steps per Episode: {SharedConfig.MAX_STEPS}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    dummy_pred_env = DummyVecEnv([make_pred_env(None) for _ in range(N_ENVS)])
    dummy_prey_env = DummyVecEnv([make_prey_env(None, None) for _ in range(N_ENVS)])
    
    # batch_size must divide evenly into n_steps * n_envs (256 * 12 = 3072)
    print("Creating NEW Predator...")
    pred_model = PPO("MlpPolicy", dummy_pred_env, verbose=0, batch_size=1536, n_steps=256, policy_kwargs=POLICY_KWARGS)

    print("Creating NEW Prey...")
    prey_model = PPO("MlpPolicy", dummy_prey_env, verbose=0, batch_size=1536, n_steps=256, policy_kwargs=POLICY_KWARGS)

    history = {
        "cycle": [],
        "avg_caught": [],
        "std_caught": [],
        "avg_steps": [],
        "avg_first_catch": [],
        "pred_training_steps": [],
        "prey_training_steps": [],
        "predator_starved": [],
        "all_prey_caught": [],
    }

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
        
        # Evaluate BEFORE prey training starts to see baseline
        baseline = evaluate_matchup(pred_model, prey_model, episodes=N_EVAL_EPISODES)
        print(f"   [Baseline before prey training: {baseline/SharedConfig.N_PREY:.1%}]")
        
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
        eval_stats = evaluate_matchup(pred_model, prey_model, episodes=N_EVAL_EPISODES, detailed=True)
        
        history["cycle"].append(cycle)
        history["avg_caught"].append(eval_stats['avg_caught'])
        history["std_caught"].append(eval_stats['std_caught'])
        history["avg_steps"].append(eval_stats['avg_steps'])
        history["avg_first_catch"].append(eval_stats['avg_first_catch'])
        history["pred_training_steps"].append(p_steps)
        history["prey_training_steps"].append(pr_steps)
        history["predator_starved"].append(eval_stats['predator_starved'])
        history["all_prey_caught"].append(eval_stats['all_prey_caught'])
        
        print(f"Result: {eval_stats['avg_caught']:.1f} ± {eval_stats['std_caught']:.1f} caught | "
              f"First catch @ step {eval_stats['avg_first_catch']:.0f} | "
              f"Starved: {eval_stats['predator_starved']}/{N_EVAL_EPISODES}")
        
        # Save models every cycle
        pred_model.save(os.path.join(MODEL_DIR, f"predator_cycle_{cycle}"))
        prey_model.save(os.path.join(MODEL_DIR, f"prey_cycle_{cycle}"))
            
        generate_gif(pred_model, prey_model, cycle)
        
        # Save history after each cycle
        with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print("Training Complete.")
    
    # Plot detailed results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Catch rate over cycles
    ax1 = axes[0, 0]
    ax1.errorbar(history["cycle"], history["avg_caught"], yerr=history["std_caught"], 
                 marker='o', capsize=3)
    ax1.set_title("Catch Rate Over Cycles")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel(f"Prey Caught (out of {SharedConfig.N_PREY})")
    ax1.grid(True, alpha=0.3)
    
    # Training effort
    ax2 = axes[0, 1]
    ax2.plot(history["cycle"], history["pred_training_steps"], marker='s', label='Predator')
    ax2.plot(history["cycle"], history["prey_training_steps"], marker='^', label='Prey')
    ax2.set_title("Training Steps Per Cycle")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Steps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Hunting efficiency
    ax3 = axes[1, 0]
    ax3.plot(history["cycle"], history["avg_first_catch"], marker='o', color='green')
    ax3.set_title("Hunting Efficiency (Steps to First Catch)")
    ax3.set_xlabel("Cycle")
    ax3.set_ylabel("Steps")
    ax3.grid(True, alpha=0.3)
    
    # Episode outcomes
    ax4 = axes[1, 1]
    ax4.bar(history["cycle"], history["all_prey_caught"], label='All Caught', alpha=0.7)
    ax4.bar(history["cycle"], history["predator_starved"], bottom=history["all_prey_caught"], 
            label='Predator Starved', alpha=0.7)
    ax4.set_title("Episode Outcomes")
    ax4.set_xlabel("Cycle")
    ax4.set_ylabel(f"Episodes (out of {N_EVAL_EPISODES})")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "coevolution_analysis.png"), dpi=150)
    print(f"Saved analysis to {OUTPUT_DIR}/coevolution_analysis.png")

if __name__ == "__main__":
    main()
