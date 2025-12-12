"""
RUN ALL - Complete training + evaluation pipeline for multiple configurations
Runs experiments with different environment settings and compares them.
"""

import os
import shutil
import subprocess
import sys
import json

COEVOLUTION_DIR = os.path.dirname(os.path.abspath(__file__))

# === EXPERIMENT CONFIGURATIONS ===
# Each config: (run_name, toroidal, obstacle_pct)
EXPERIMENTS = [
    ("toroidal_with_obstacles", True, 0.05),
    ("toroidal_no_obstacles", True, 0.0),
    ("bounded_with_obstacles", False, 0.05),
    ("bounded_no_obstacles", False, 0.0),
]

def update_env_config(toroidal: bool, obstacle_pct: float):
    """Modify the environment file to use specified settings."""
    env_file = os.path.join(COEVOLUTION_DIR, "predator_prey_multi_env.py")
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Update toroidal setting in _move_entities
    if toroidal:
        # Wrap around (toroidal)
        content = content.replace(
            "new_pos = np.clip(new_pos, 0, self.grid_size - 1)",
            "new_pos = new_pos % self.grid_size"
        )
    else:
        # Bounded (clip to edges)
        content = content.replace(
            "new_pos = new_pos % self.grid_size",
            "new_pos = np.clip(new_pos, 0, self.grid_size - 1)"
        )
    
    # Update obstacle percentage
    import re
    # Find and replace the obstacle generation line
    old_pattern = r"num_obs = int\(self\.grid_size \* self\.grid_size \* [\d.]+\)"
    new_line = f"num_obs = int(self.grid_size * self.grid_size * {obstacle_pct})"
    content = re.sub(old_pattern, new_line, content)
    
    with open(env_file, 'w') as f:
        f.write(content)

def update_run_name(run_name: str):
    """Update RUN_NAME in train_coevolution.py"""
    train_file = os.path.join(COEVOLUTION_DIR, "train_coevolution.py")
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    import re
    content = re.sub(
        r'RUN_NAME = "[^"]*"',
        f'RUN_NAME = "{run_name}"',
        content
    )
    
    with open(train_file, 'w') as f:
        f.write(content)

def run_experiment(run_name: str, toroidal: bool, obstacle_pct: float):
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {run_name}")
    print(f"  Toroidal: {toroidal}")
    print(f"  Obstacles: {obstacle_pct*100:.0f}%")
    print(f"{'='*60}")
    
    # Clear old data for this run
    models_dir = os.path.join(COEVOLUTION_DIR, "models", run_name)
    output_dir = os.path.join(COEVOLUTION_DIR, "output", run_name)
    
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Update configs
    update_env_config(toroidal, obstacle_pct)
    update_run_name(run_name)
    
    # Run training
    print("\n>> Training...")
    train_script = os.path.join(COEVOLUTION_DIR, "train_coevolution.py")
    result = subprocess.run([sys.executable, train_script], cwd=COEVOLUTION_DIR)
    
    if result.returncode != 0:
        print(f"Training failed for {run_name}!")
        return False
    
    # Run evaluation
    print("\n>> Evaluating...")
    eval_script = os.path.join(COEVOLUTION_DIR, "evaluate.py")
    result = subprocess.run([sys.executable, eval_script, run_name], cwd=COEVOLUTION_DIR)
    
    if result.returncode != 0:
        print(f"Evaluation failed for {run_name}!")
        return False
    
    print(f"\n>> {run_name} complete!")
    return True

def generate_comparison_summary():
    """Generate a summary comparing all experiments."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    results = {}
    for run_name, toroidal, obs_pct in EXPERIMENTS:
        history_file = os.path.join(COEVOLUTION_DIR, "output", run_name, "history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            if history.get("avg_caught"):
                final_catch = history["avg_caught"][-1]
                results[run_name] = {
                    "toroidal": toroidal,
                    "obstacles": obs_pct,
                    "final_catch_rate": final_catch / 50 * 100,
                    "cycles": len(history["avg_caught"])
                }
    
    print(f"\n{'Experiment':<30} {'Toroidal':<10} {'Obstacles':<10} {'Final Catch%':<12}")
    print("-" * 62)
    for name, data in results.items():
        print(f"{name:<30} {str(data['toroidal']):<10} {data['obstacles']*100:>6.0f}%    {data['final_catch_rate']:>8.1f}%")
    
    # Save comparison
    summary_file = os.path.join(COEVOLUTION_DIR, "output", "comparison_summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {summary_file}")

def main():
    print("="*60)
    print("MULTI-EXPERIMENT PIPELINE")
    print(f"Running {len(EXPERIMENTS)} configurations...")
    print("="*60)
    
    successful = []
    failed = []
    
    for run_name, toroidal, obstacle_pct in EXPERIMENTS:
        if run_experiment(run_name, toroidal, obstacle_pct):
            successful.append(run_name)
        else:
            failed.append(run_name)
    
    # Generate comparison
    generate_comparison_summary()
    
    # Final summary
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"\nSuccessful: {len(successful)}/{len(EXPERIMENTS)}")
    for name in successful:
        print(f"  ✓ {name}")
    if failed:
        print(f"\nFailed: {len(failed)}")
        for name in failed:
            print(f"  ✗ {name}")
    
    print(f"\nOutputs in: {os.path.join(COEVOLUTION_DIR, 'output')}")
    print(f"Models in: {os.path.join(COEVOLUTION_DIR, 'models')}")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
