"""
RUN ALL - Complete training + evaluation pipeline for multiple configurations
Runs experiments in PARALLEL for maximum speed.
"""

import os
import shutil
import subprocess
import sys
import time
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

def generate_comparison_summary():
    """Generate a summary comparing all experiments."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    results = {}
    for run_name, toroidal, obs_pct in EXPERIMENTS:
        history_file = os.path.join(COEVOLUTION_DIR, "output", run_name, "history.json")
        if os.path.exists(history_file):
            try:
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
            except Exception as e:
                print(f"Error reading history for {run_name}: {e}")
    
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

def run_experiments_parallel():
    # 1. TRAINING PHASE
    training_processes = []
    print(f"Launching {len(EXPERIMENTS)} TRAINING experiments in parallel...")
    
    for run_name, toroidal, obstacle_pct in EXPERIMENTS:
        print(f"Preparing: {run_name} (Toroidal={toroidal}, Obstacles={obstacle_pct})")
        
        # Clear old data
        models_dir = os.path.join(COEVOLUTION_DIR, "models", run_name)
        output_dir = os.path.join(COEVOLUTION_DIR, "output", run_name)
        
        if os.path.exists(models_dir):
            try: shutil.rmtree(models_dir)
            except: pass
        if os.path.exists(output_dir):
            try: shutil.rmtree(output_dir)
            except: pass
            
        # Prepare Env Vars
        env = os.environ.copy()
        env["RUN_NAME"] = run_name
        env["ENV_TOROIDAL"] = str(toroidal)
        env["ENV_OBSTACLE_PCT"] = str(obstacle_pct)
        env["N_ENVS"] = "4" 
        env["SKIP_GIFS"] = "False"
        
        train_script = os.path.join(COEVOLUTION_DIR, "train_coevolution.py")
        log_file = os.path.join(COEVOLUTION_DIR, f"{run_name}_train.log")
        
        with open(log_file, "w") as f:
            p = subprocess.Popen(
                [sys.executable, train_script], 
                cwd=COEVOLUTION_DIR,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT
            )
        training_processes.append((run_name, p, env))
        print(f"Started TRAINING {run_name} (PID: {p.pid}) - Logging to {log_file}")
        
    print(f"\nAll training experiments running. Please wait...")
    
    # Wait for training completion
    successful_runs = []
    for run_name, p, env in training_processes:
        p.wait()
        if p.returncode != 0:
            print(f"Training {run_name} FAILED with code {p.returncode}. Check log.")
        else:
            print(f"Training {run_name} COMPLETED.")
            successful_runs.append((run_name, env))

    # 2. EVALUATION PHASE
    print(f"\nStarting EVALUATION for {len(successful_runs)} successful runs...")
    eval_processes = []
    
    for run_name, env in successful_runs:
        eval_script = os.path.join(COEVOLUTION_DIR, "evaluate.py")
        log_file = os.path.join(COEVOLUTION_DIR, f"{run_name}_eval.log")
        
        # Evaluate needs the same env vars to configure the environment correctly
        with open(log_file, "w") as f:
            p = subprocess.Popen(
                [sys.executable, eval_script, run_name], 
                cwd=COEVOLUTION_DIR,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT
            )
        eval_processes.append((run_name, p))
        print(f"Started EVALUATION {run_name} (PID: {p.pid})")

    # Wait for evaluation completion
    for run_name, p in eval_processes:
        p.wait()
        if p.returncode != 0:
            print(f"Evaluation {run_name} FAILED.")
        else:
            print(f"Evaluation {run_name} COMPLETED.")

    # 3. SUMMARY
    generate_comparison_summary()

if __name__ == "__main__":
    run_experiments_parallel()
