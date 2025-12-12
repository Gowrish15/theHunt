"""
Cross-cycle evaluation script for co-evolution analysis.
Tests trained models against each other and random baselines.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from stable_baselines3 import PPO
from predator_prey_multi_env import PredatorTrainingEnv, PreyTrainingEnv, SharedConfig

MODEL_DIR = "models"
OUTPUT_DIR = "output"
N_EVAL_EPISODES = 20
RUN_NAME = None  # Set via command line or manually


def set_run(name):
    """Set the run to evaluate."""
    global MODEL_DIR, OUTPUT_DIR, RUN_NAME
    RUN_NAME = name
    MODEL_DIR = os.path.join("models", name)
    OUTPUT_DIR = os.path.join("output", name)


def load_models(cycle):
    """Load predator and prey models from a specific cycle."""
    pred_path = os.path.join(MODEL_DIR, f"predator_cycle_{cycle}.zip")
    prey_path = os.path.join(MODEL_DIR, f"prey_cycle_{cycle}.zip")
    
    pred = PPO.load(pred_path) if os.path.exists(pred_path) else None
    prey = PPO.load(prey_path) if os.path.exists(prey_path) else None
    return pred, prey


def evaluate(pred_model, prey_model, episodes=N_EVAL_EPISODES):
    """Run evaluation, prey_model can be None for random baseline."""
    env = PredatorTrainingEnv(prey_policy=prey_model)
    
    catches = []
    steps_list = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            if pred_model:
                action, _ = pred_model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        
        catches.append(info['total_caught'])
        steps_list.append(env.steps)
    
    return {
        'avg_caught': np.mean(catches),
        'std_caught': np.std(catches),
        'catch_rate': np.mean(catches) / SharedConfig.N_PREY,
        'avg_steps': np.mean(steps_list),
    }


def get_available_cycles():
    """Find all cycles that have saved models."""
    cycles = set()
    if not os.path.exists(MODEL_DIR):
        return []
    for f in os.listdir(MODEL_DIR):
        if f.startswith("predator_cycle_"):
            try:
                c = int(f.replace("predator_cycle_", "").replace(".zip", ""))
                cycles.add(c)
            except ValueError:
                pass
    return sorted(cycles)


def run_baseline_comparison():
    """Compare trained models vs random baselines."""
    cycles = get_available_cycles()
    if not cycles:
        print("No saved models found!")
        return
    
    print("=== BASELINE COMPARISON ===")
    print(f"Found models for cycles: {cycles}")
    
    results = []
    
    for cycle in cycles:
        pred, prey = load_models(cycle)
        
        # Trained pred vs trained prey
        trained_vs_trained = evaluate(pred, prey)
        
        # Trained pred vs random prey
        trained_vs_random = evaluate(pred, None)
        
        # Random pred vs trained prey
        random_vs_trained = evaluate(None, prey)
        
        results.append({
            'cycle': cycle,
            'trained_vs_trained': trained_vs_trained['catch_rate'],
            'trained_vs_random': trained_vs_random['catch_rate'],
            'random_vs_trained': random_vs_trained['catch_rate'],
        })
        
        print(f"\nCycle {cycle}:")
        print(f"  Trained Pred vs Trained Prey: {trained_vs_trained['catch_rate']:.1%}")
        print(f"  Trained Pred vs Random Prey:  {trained_vs_random['catch_rate']:.1%}")
        print(f"  Random Pred vs Trained Prey:  {random_vs_trained['catch_rate']:.1%}")
    
    return results


def run_cross_cycle_tournament():
    """Test all predators against all prey versions."""
    cycles = get_available_cycles()
    if len(cycles) < 2:
        print("Need at least 2 cycles for tournament!")
        return
    
    print("\n=== CROSS-CYCLE TOURNAMENT ===")
    
    # Load all models
    models = {}
    for c in cycles:
        pred, prey = load_models(c)
        models[c] = {'pred': pred, 'prey': prey}
    
    # Build matchup matrix
    n = len(cycles)
    matrix = np.zeros((n, n))
    
    print("\nRunning matchups...")
    for i, pred_cycle in enumerate(cycles):
        for j, prey_cycle in enumerate(cycles):
            result = evaluate(models[pred_cycle]['pred'], models[prey_cycle]['prey'])
            matrix[i, j] = result['catch_rate']
            print(f"  Pred C{pred_cycle} vs Prey C{prey_cycle}: {result['catch_rate']:.1%}")
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Prey C{c}" for c in cycles])
    ax.set_yticklabels([f"Pred C{c}" for c in cycles])
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", 
                          color="black" if 0.3 < matrix[i, j] < 0.7 else "white")
    
    ax.set_title("Cross-Cycle Tournament (Catch Rate)")
    ax.set_xlabel("Prey Generation")
    ax.set_ylabel("Predator Generation")
    
    plt.colorbar(im, label="Catch Rate")
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "tournament_matrix.png"), dpi=150)
    print(f"\nSaved tournament matrix to {OUTPUT_DIR}/tournament_matrix.png")
    
    return matrix, cycles


def analyze_progression():
    """Analyze if later generations are actually better."""
    cycles = get_available_cycles()
    if len(cycles) < 3:
        print("Need at least 3 cycles for progression analysis!")
        return
    
    print("\n=== PROGRESSION ANALYSIS ===")
    
    models = {}
    for c in cycles:
        pred, prey = load_models(c)
        models[c] = {'pred': pred, 'prey': prey}
    
    # Test each predator against cycle 1 prey (baseline)
    print("\nPredator improvement (vs Cycle 1 prey):")
    pred_progress = []
    for c in cycles:
        result = evaluate(models[c]['pred'], models[1]['prey'])
        pred_progress.append(result['catch_rate'])
        print(f"  Predator C{c}: {result['catch_rate']:.1%}")
    
    # Test each prey against cycle 1 predator (baseline)
    print("\nPrey improvement (vs Cycle 1 predator):")
    prey_progress = []
    for c in cycles:
        result = evaluate(models[1]['pred'], models[c]['prey'])
        # Lower catch rate = better prey
        prey_progress.append(1 - result['catch_rate'])
        print(f"  Prey C{c} survival: {1 - result['catch_rate']:.1%}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cycles, pred_progress, marker='o', label='Predator skill (vs C1 prey)')
    ax.plot(cycles, prey_progress, marker='s', label='Prey survival (vs C1 pred)')
    ax.set_xlabel("Training Cycle")
    ax.set_ylabel("Performance")
    ax.set_title("Agent Progression Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "progression_analysis.png"), dpi=150)
    print(f"\nSaved progression analysis to {OUTPUT_DIR}/progression_analysis.png")


def generate_gif(pred_model, prey_model, filename, title="Replay", max_frames=500):
    """Generate a GIF for a specific matchup."""
    env = PredatorTrainingEnv(prey_policy=prey_model)
    obs, _ = env.reset()
    
    frames = []
    done = False
    step_count = 0
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.tight_layout(pad=0.5)
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Draw obstacles
    obs_y, obs_x = np.where(env.obstacles)
    if len(obs_x) > 0:
        ax.scatter(obs_x, obs_y, c='gray', s=30, marker='s', alpha=0.5)
    
    predator_scatter = ax.scatter([], [], c='red', s=100, marker='x')
    prey_scatter = ax.scatter([], [], c='blue', s=50)
    title_text = ax.set_title(title)
    
    # Positions are [row,col]=[y,x], scatter wants [x,y]
    predator_scatter.set_offsets([[env.pred[1], env.pred[0]]])
    if np.any(env.prey_alive):
        alive_prey = env.prey[env.prey_alive]
        prey_scatter.set_offsets(alive_prey[:, ::-1])
    
    fig.canvas.draw()
    expected_shape = np.asarray(fig.canvas.buffer_rgba()).shape[:2]
    
    while not done and step_count < max_frames:
        step_count += 1
        title_text.set_text(f"{title} | Step {step_count} | Caught: {env.caught_total}/{env.n_prey}")
        
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        if buf.shape[:2] == expected_shape:
            frames.append(buf[:, :, :3].copy())
        
        if pred_model:
            action, _ = pred_model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        
        # Swap [row,col] to [x,y] for scatter plot
        predator_scatter.set_offsets([[env.pred[1], env.pred[0]]])
        if np.any(env.prey_alive):
            alive_prey = env.prey[env.prey_alive]
            prey_scatter.set_offsets(alive_prey[:, ::-1])
        else:
            prey_scatter.set_offsets(np.empty((0, 2)))
    
    plt.close(fig)
    
    if frames:
        imageio.mimsave(filename, frames, fps=10, loop=0)
        print(f"  Saved: {filename}")


def generate_highlight_gifs():
    """Generate GIFs for each cycle's matchup."""
    cycles = get_available_cycles()
    if not cycles:
        return
    
    print("\n=== GENERATING CYCLE GIFS ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    models = {}
    for c in cycles:
        pred, prey = load_models(c)
        models[c] = {'pred': pred, 'prey': prey}
    
    for cycle in cycles:
        print(f"Generating Cycle {cycle} replay...")
        generate_gif(
            models[cycle]['pred'], 
            models[cycle]['prey'],
            os.path.join(OUTPUT_DIR, f"cycle_{cycle}.gif"),
            f"Cycle {cycle}"
        )


def main():
    import sys
    
    # Get run name from command line or use default
    if len(sys.argv) > 1:
        set_run(sys.argv[1])
    else:
        # List available runs
        if os.path.exists("models"):
            runs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
            if runs:
                print("Available runs:", runs)
                print(f"Usage: python evaluate.py <run_name>")
                print(f"\nUsing first available: {runs[0]}")
                set_run(runs[0])
            else:
                print("No runs found in models/")
                return
        else:
            print("No models directory found!")
            return
    
    print("=" * 60)
    print(f"CO-EVOLUTION EVALUATION: {RUN_NAME}")
    print("=" * 60)
    
    baseline_results = run_baseline_comparison()
    
    if baseline_results and len(baseline_results) >= 2:
        run_cross_cycle_tournament()
        analyze_progression()
        generate_highlight_gifs()
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
