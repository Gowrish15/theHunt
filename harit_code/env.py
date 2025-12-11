import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RefinedPredatorPreyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, grid_size: int = 30, n_prey: int = 50, max_steps: int = 400):
        super().__init__()
        self.grid_size = grid_size
        self.n_prey = n_prey
        self.max_steps = max_steps
        self.difficulty = 0.0

        # Pre-calculate move deltas for fast indexing
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
        self.move_deltas = np.array([
            [0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]
        ], dtype=np.int32)

        # Create static obstacles
        self.obstacles = np.zeros((grid_size, grid_size), dtype=bool)
        rng = np.random.default_rng(42)
        num_obs = int(grid_size * grid_size * 0.05)
        obs_x = rng.integers(0, grid_size, num_obs)
        obs_y = rng.integers(0, grid_size, num_obs)
        self.obstacles[obs_x, obs_y] = True

        low = np.array([0.0]*2 + [-grid_size]*10 + [0.0], dtype=np.float32)
        high = np.array([grid_size]*2 + [grid_size]*10 + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self._rng = np.random.default_rng()

    def set_difficulty(self, value: float):
        self.difficulty = float(np.clip(value, 0.0, 1.0))

    def _get_obs(self):
        # Optimized observation calculation
        if not np.any(self.prey_alive):
            vecs = np.zeros(10, dtype=np.float32)
        else:
            # Vectorized distance calculation
            valid_prey = self.prey[self.prey_alive]
            diffs = valid_prey - self.pred
            dists_sq = np.sum(diffs**2, axis=1) # Avoid sqrt for sorting speed
            
            # Get nearest 5 using partition (faster than full sort)
            k = min(5, len(dists_sq))
            nearest_indices = np.argpartition(dists_sq, k-1)[:k]
            nearest_diffs = diffs[nearest_indices]
            
            # Sort the top k precisely
            nearest_dists = dists_sq[nearest_indices]
            sorted_order = np.argsort(nearest_dists)
            nearest_diffs = nearest_diffs[sorted_order]
            
            if k < 5:
                pad = np.zeros((5 - k, 2), dtype=np.float32)
                nearest_diffs = np.vstack([nearest_diffs, pad])
            
            vecs = nearest_diffs.flatten()

        alive_frac = np.mean(self.prey_alive) if self.n_prey > 0 else 0.0
        return np.concatenate([self.pred, vecs, [alive_frac]]).astype(np.float32)

    def _move_entities(self, positions, actions):
        """Vectorized movement logic for multiple entities."""
        # Calculate potential new positions
        deltas = self.move_deltas[actions]
        new_pos = positions + deltas
        
        # Check boundaries (vectorized)
        in_bounds_x = (new_pos[:, 0] >= 0) & (new_pos[:, 0] < self.grid_size)
        in_bounds_y = (new_pos[:, 1] >= 0) & (new_pos[:, 1] < self.grid_size)
        valid_bounds = in_bounds_x & in_bounds_y
        
        # Check obstacles (only for those inside bounds)
        # We clamp to avoid index errors, but use the mask to reject invalid ones
        safe_x = np.clip(new_pos[:, 0], 0, self.grid_size-1)
        safe_y = np.clip(new_pos[:, 1], 0, self.grid_size-1)
        hit_obstacle = self.obstacles[safe_x, safe_y]
        
        can_move = valid_bounds & (~hit_obstacle)
        
        # Update only valid moves
        # where(condition, x, y)
        final_pos = np.where(can_move[:, None], new_pos, positions)
        return final_pos.astype(np.int32)

    def _prey_logic(self):
        vision_radius_sq = (2.0 + (self.difficulty * 10.0))**2
        
        # 1. Default Random Actions
        acts = self._rng.integers(0, 5, size=self.n_prey)

        # 2. Smart Logic (Vectorized)
        if self.difficulty > 0:
            # Identify smart prey
            is_smart = self._rng.random(self.n_prey) < 0.8
            active_smart_mask = self.prey_alive & is_smart
            
            if np.any(active_smart_mask):
                smart_indices = np.where(active_smart_mask)[0]
                smart_pos = self.prey[smart_indices]
                
                # Calculate distance to predator
                dists_sq = np.sum((smart_pos - self.pred)**2, axis=1)
                flee_mask = dists_sq < vision_radius_sq
                
                # Only process those that need to flee
                flee_indices = smart_indices[flee_mask]
                
                if len(flee_indices) > 0:
                    flee_pos = self.prey[flee_indices]
                    
                    # Evaluate all 4 directions for these prey
                    # Shape: (N_flee, 4 directions, 2 coords)
                    # We broadcast add: (N,1,2) + (1,4,2)
                    candidates = flee_pos[:, None, :] + self.move_deltas[:4][None, :, :]
                    
                    # Calculate dists to predator for all candidates
                    # (N, 4, 2) - (2) -> (N, 4, 2)
                    cand_dists = np.sum((candidates - self.pred)**2, axis=2)
                    
                    # Find best action (max distance)
                    best_actions = np.argmax(cand_dists, axis=1)
                    
                    # Assign back
                    acts[flee_indices] = best_actions

        return acts

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.caught_total = 0
        
        # Fast random spawn avoiding obstacles
        # We generate more points than needed and filter
        candidates = self._rng.integers(0, self.grid_size, size=(self.n_prey + 10, 2))
        valid_mask = ~self.obstacles[candidates[:, 0], candidates[:, 1]]
        valid_points = candidates[valid_mask]
        
        self.pred = valid_points[0]
        self.prey = valid_points[1 : self.n_prey + 1]
        
        # If we didn't get enough valid points (rare), fallback to slow init
        if len(self.prey) < self.n_prey:
             # Just fill the rest randomly (simplification for speed)
             padding = self._rng.integers(0, self.grid_size, size=(self.n_prey - len(self.prey), 2))
             self.prey = np.vstack([self.prey, padding])

        self.prey_alive = np.ones(self.n_prey, dtype=bool)
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        
        # 1. Move Predator
        # Wrap single action in array for vectorized function
        pred_pos_arr = self.pred[None, :]
        new_pred_pos = self._move_entities(pred_pos_arr, np.array([action]))
        self.pred = new_pred_pos[0]
        
        # 2. Move Prey (Vectorized)
        prey_acts = self._prey_logic()
        self.prey = self._move_entities(self.prey, prey_acts)

        # 3. Check Catches (Vectorized)
        # Broadcasting: (N, 2) == (2)
        caught_mask = np.all(self.prey == self.pred, axis=1)
        new_catches = np.sum(caught_mask & self.prey_alive)
        self.prey_alive[caught_mask] = False
        self.caught_total += new_catches

        reward = float(new_catches) * 10.0 - 0.1
        terminated = not np.any(self.prey_alive)
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {"total_caught": self.caught_total}