import itertools
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from env import RefinedPredatorPreyEnv

# === CONFIGURATION ===
N_ENVS = 8  # Use 8 parallel environments
TRAIN_BUDGET = 60_000  # Steps per candidate (keep low for speed)

def run_grid_search():
    # 1. Hardware Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Grid Search running on: {device.upper()}")

    # 2. Define Hyperparameter Grid
    param_grid = {
        'learning_rate': [1e-3, 3e-4, 1e-4],  # Aggressive vs Precise
        'gamma': [0.95, 0.99],                # Short-term vs Long-term strategy
        'n_steps': [1024, 2048],              # Batch size per update
        'ent_coef': [0.0, 0.01]               # Exploration bonus
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    print(f"Testing {len(combinations)} candidates...")

    # 3. Main Search Loop
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        # Create Parallel Envs for Training
        # We use a fixed difficulty (0.3) to make comparisons fair
        train_env = make_vec_env(
            RefinedPredatorPreyEnv, 
            n_envs=N_ENVS, 
            vec_env_cls=SubprocVecEnv,
            env_kwargs={'grid_size': 20, 'n_prey': 40} # Standard settings
        )
        # Set difficulty on all envs
        train_env.env_method("set_difficulty", 0.3)

        try:
            # Initialize Model
            model = PPO(
                "MlpPolicy", 
                train_env, 
                verbose=0,
                device=device,
                batch_size=2048, # Optimized for RTX 3060
                **params
            )
            
            # Train
            model.learn(total_timesteps=TRAIN_BUDGET)
            
            # Evaluate (using a fresh single env for accuracy)
            eval_env = RefinedPredatorPreyEnv(grid_size=20, n_prey=40)
            eval_env.set_difficulty(0.3)
            
            scores = []
            for _ in range(10): # 10 Test Episodes
                obs, _ = eval_env.reset()
                done = False
                info = {}
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = eval_env.step(action)
                    done = term or trunc
                
                # Metric: Catch Rate (0.0 to 1.0)
                scores.append(info.get("total_caught", 0) / 40.0)
            
            avg_score = np.mean(scores)
            print(f" -> Result: Catch Rate = {avg_score:.2%}")
            
            results.append({**params, 'score': avg_score})

        except Exception as e:
            print(f"Candidate failed: {e}")
        finally:
            train_env.close()

    # 4. Save and Show Results
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='score', ascending=False)
        print("\n=== TOP 5 CONFIGURATIONS ===")
        print(df.head(5))
        
        best_params = df.iloc[0].to_dict()
        print(f"\nBest Parameters found: {best_params}")
        df.to_csv("grid_search_results.csv", index=False)
    else:
        print("No results found.")

if __name__ == "__main__":
    run_grid_search()