import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SharedConfig:
    """Configuration shared between both environments"""
    GRID_SIZE = 30
    N_PREY = 50
    MAX_STEPS = 1000
    PREY_SPEED_MODIFIER = 2  # Prey skips move every 2nd step (50% speed)
    
    # Energy System
    PREDATOR_START_ENERGY = 100.0
    PREDATOR_MAX_ENERGY = 200.0
    PREDATOR_STEP_COST = 0.2
    PREDATOR_EAT_GAIN = 40.0
    
    PREY_START_ENERGY = 50.0
    PREY_MAX_ENERGY = 100.0
    PREY_MOVE_COST = 0.5
    PREY_REST_GAIN = 1.0

class BasePredatorPreyEnv(gym.Env):
    """Base class with shared physics and logic"""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        self.grid_size = SharedConfig.GRID_SIZE
        self.n_prey = SharedConfig.N_PREY
        self.max_steps = SharedConfig.MAX_STEPS
        
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
        self.move_deltas = np.array([[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]], dtype=np.int32)
        
        self.observation_space = None 
        self.action_space = spaces.Discrete(5)
        
        self._rng = np.random.default_rng()
        self.obstacles = self._generate_obstacles()
        
        # Energy State
        self.pred_energy = SharedConfig.PREDATOR_START_ENERGY
        self.prey_energy = np.full(self.n_prey, SharedConfig.PREY_START_ENERGY, dtype=np.float32)

    def _generate_obstacles(self):
        obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        num_obs = int(self.grid_size * self.grid_size * 0.05)
        obs_x = self._rng.integers(0, self.grid_size, num_obs)
        obs_y = self._rng.integers(0, self.grid_size, num_obs)
        obstacles[obs_x, obs_y] = True
        return obstacles

    def _move_entities(self, positions, actions):
        """Vectorized movement logic with Wrap-Around (Toroidal)"""
        deltas = self.move_deltas[actions]
        new_pos = positions + deltas
        
        # Wrap around (0 -> 29, 29 -> 0)
        new_pos = new_pos % self.grid_size
        
        # Check obstacles (if any exist)
        hit_obstacle = self.obstacles[new_pos[:, 0], new_pos[:, 1]]
        
        can_move = ~hit_obstacle
        return np.where(can_move[:, None], new_pos, positions).astype(np.int32)

    def _get_toroidal_diff(self, pos1, pos2):
        """Calculate shortest vector from pos1 to pos2 on a torus"""
        diff = pos2 - pos1
        half = self.grid_size / 2.0
        # If diff > half, subtract size (wrap other way)
        # If diff < -half, add size
        diff = (diff + half) % self.grid_size - half
        return diff

    def _get_min_prey_dist(self):
        """Helper to get distance to nearest alive prey"""
        if not np.any(self.prey_alive):
            return 0.0
        valid_prey = self.prey[self.prey_alive]
        
        # Use toroidal difference
        diffs = self._get_toroidal_diff(self.pred, valid_prey)
        dists = np.linalg.norm(diffs, axis=1)
        return np.min(dists)

    def _get_obstacle_sensor(self, pos):
        """Raycast in 8 directions to find distance to obstacles/walls"""
        # Directions: N, NE, E, SE, S, SW, W, NW
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1), 
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        sensors = np.zeros(8, dtype=np.float32)
        
        for i, (dx, dy) in enumerate(directions):
            dist = 0.0
            x, y = pos[0], pos[1]
            
            # Raycast
            for _ in range(self.grid_size): 
                x += dx
                y += dy
                dist += 1.0
                
                # Check bounds
                if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                    break # Hit wall
                
                # Check obstacle
                if self.obstacles[int(x), int(y)]:
                    break # Hit obstacle
            
            sensors[i] = dist / self.grid_size
            
        return sensors

    def _get_predator_obs(self):
        """Predator sees: Self pos, 5 nearest prey vectors, prey alive count, 8 obstacle sensors"""
        if not np.any(self.prey_alive):
            vecs = np.zeros(10, dtype=np.float32)
        else:
            valid_prey = self.prey[self.prey_alive]
            
            # Use toroidal difference
            diffs = self._get_toroidal_diff(self.pred, valid_prey)
            dists_sq = np.sum(diffs**2, axis=1)
            
            k = min(5, len(dists_sq))
            nearest_indices = np.argpartition(dists_sq, k-1)[:k]
            nearest_diffs = diffs[nearest_indices]
            
            # Sort by distance
            nearest_dists = dists_sq[nearest_indices]
            sorted_order = np.argsort(nearest_dists)
            nearest_diffs = nearest_diffs[sorted_order]
            
            if k < 5:
                pad = np.zeros((5 - k, 2), dtype=np.float32)
                nearest_diffs = np.vstack([nearest_diffs, pad])
            
            vecs = nearest_diffs.flatten()

        alive_frac = np.mean(self.prey_alive) if self.n_prey > 0 else 0.0
        sensors = self._get_obstacle_sensor(self.pred)
        
        # Normalize energy to 0-1
        energy_norm = self.pred_energy / SharedConfig.PREDATOR_MAX_ENERGY
        
        return np.concatenate([self.pred, vecs, [alive_frac], sensors, [energy_norm]]).astype(np.float32)

    def _get_prey_obs(self, prey_idx):
        """
        Prey sees: 
        1. Self Pos (2)
        2. Vector to Predator (2)
        3. Vector to Nearest Other Prey (2)
        4. Self Energy (1)
        """
        my_pos = self.prey[prey_idx]
        
        # 1. Vector to Predator (Toroidal)
        to_pred = self._get_toroidal_diff(my_pos, self.pred)
        
        # 2. Vector to Nearest Other Prey (Flocking)
        other_prey_mask = self.prey_alive.copy()
        other_prey_mask[prey_idx] = False # Don't count self
        
        if np.any(other_prey_mask):
            others = self.prey[other_prey_mask]
            
            # Toroidal diffs to all others
            diffs = self._get_toroidal_diff(my_pos, others)
            dists_sq = np.sum(diffs**2, axis=1)
            nearest_idx = np.argmin(dists_sq)
            to_friend = diffs[nearest_idx]
        else:
            to_friend = np.zeros(2, dtype=np.float32)
            
        # 3. Energy
        energy_norm = self.prey_energy[prey_idx] / SharedConfig.PREY_MAX_ENERGY
            
        return np.concatenate([my_pos, to_pred, to_friend, [energy_norm]]).astype(np.float32)

# ==========================================
# 1. PREDATOR TRAINING ENV
# ==========================================
class PredatorTrainingEnv(BasePredatorPreyEnv):
    """
    Agent: Predator
    Opponent: Prey (Controlled by a loaded Policy)
    """
    def __init__(self, prey_policy=None):
        super().__init__()
        # Observation: Pred Pos (2) + 5 Nearest Prey (10) + Alive Frac (1) + 8 Sensors + Energy (1) = 22
        low = np.array([0.0]*22, dtype=np.float32)
        high = np.array([self.grid_size]*22, dtype=np.float32)
        # Fix ranges for relative vectors (can be negative)
        low[2:12] = -self.grid_size
        high[2:12] = self.grid_size
        # Alive frac
        low[12] = 0.0
        high[12] = 1.0
        # Sensors (0-1)
        low[13:21] = 0.0
        high[13:21] = 1.0
        # Energy (0-1)
        low[21] = 0.0
        high[21] = 1.0
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.prey_policy = prey_policy

    def set_prey_policy(self, policy):
        self.prey_policy = policy

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.caught_total = 0
        
        # Spawn logic
        candidates = self._rng.integers(0, self.grid_size, size=(self.n_prey + 10, 2))
        valid_mask = ~self.obstacles[candidates[:, 0], candidates[:, 1]]
        valid_points = candidates[valid_mask]
        
        if len(valid_points) < self.n_prey + 1:
             valid_points = np.zeros((self.n_prey + 1, 2), dtype=np.int32)
        
        self.pred = valid_points[0]
        self.prey = valid_points[1 : self.n_prey + 1]
        self.prey_alive = np.ones(self.n_prey, dtype=bool)
        
        return self._get_predator_obs(), {}

    def step(self, action):
        self.steps += 1
        
        # Calculate distance BEFORE move
        dist_before = self._get_min_prey_dist()
        old_pred_pos = self.pred.copy()
        
        # 1. Move Predator (Agent)
        self.pred = self._move_entities(self.pred[None, :], np.array([action]))[0]
        
        # Check if stuck (didn't move but tried to move)
        moved = not np.array_equal(self.pred, old_pred_pos)
        tried_to_move = action != 4 # 4 is Stay
        
        # Energy Logic (Predator)
        if tried_to_move:
            self.pred_energy -= SharedConfig.PREDATOR_STEP_COST
        
        # Calculate distance AFTER move
        dist_after = self._get_min_prey_dist()
        
        # 2. Move Prey (Opponent AI)
        # Speed Bias: Prey skips every Nth turn
        if self.steps % SharedConfig.PREY_SPEED_MODIFIER != 0:
            if self.prey_policy:
                # Construct batch obs for ALL prey
                # Obs shape: (N_Prey, 7) -> [Pos, ToPred, ToFriend, Energy]
                
                # Vectorized construction
                to_pred = self._get_toroidal_diff(self.prey, self.pred)
                
                # Simplified "To Friend" (Nearest neighbor is expensive to compute for all 50 every step)
                # For now, we'll use zero to avoid bad signals.
                to_friend = np.zeros_like(self.prey)
                
                # Energy
                energy_norm = (self.prey_energy / SharedConfig.PREY_MAX_ENERGY)[:, None]
                
                prey_obs = np.hstack([self.prey, to_pred, to_friend, energy_norm]).astype(np.float32)
                
                # Predict actions for all prey at once
                prey_actions, _ = self.prey_policy.predict(prey_obs, deterministic=False)
            else:
                # Random fallback
                prey_actions = self._rng.integers(0, 5, size=self.n_prey)
            
            # Energy Logic (Prey)
            # Moving costs energy, Staying rests
            moving_mask = prey_actions != 4
            self.prey_energy[moving_mask] -= SharedConfig.PREY_MOVE_COST
            self.prey_energy[~moving_mask] += SharedConfig.PREY_REST_GAIN
            self.prey_energy = np.clip(self.prey_energy, 0, SharedConfig.PREY_MAX_ENERGY)
            
            # Starvation Check
            starved_mask = self.prey_energy <= 0
            self.prey_alive[starved_mask] = False
            
            self.prey = self._move_entities(self.prey, prey_actions)

        # 3. Check Catches
        caught_mask = np.all(self.prey == self.pred, axis=1)
        new_catches = np.sum(caught_mask & self.prey_alive)
        self.prey_alive[caught_mask] = False
        
        # Predator gains energy for eating
        if new_catches > 0:
            self.pred_energy += new_catches * SharedConfig.PREDATOR_EAT_GAIN
            self.pred_energy = min(self.pred_energy, SharedConfig.PREDATOR_MAX_ENERGY)
        
        # Progressive Reward: 1st=10, 2nd=20, 3rd=30...
        # This encourages catching ALL prey, not just farming or stopping.
        reward = 0.0
        if new_catches > 0:
            for i in range(new_catches):
                reward += (self.caught_total + 1 + i) * 10.0
        
        self.caught_total += new_catches

        # Time penalty (Reduced to allow exploration)
        reward -= 0.1
        
        # Shaping: Reward for getting closer (Stronger scent to guide on large map)
        reward += (dist_before - dist_after) * 0.5
        
        # Penalty for hitting obstacles/walls (Stuck)
        if tried_to_move and not moved:
            reward -= 0.5
            
        terminated = False
        # Predator Starvation Penalty
        if self.pred_energy <= 0:
            reward -= 10.0 # Big penalty for starving
            terminated = True
        
        terminated = terminated or (not np.any(self.prey_alive))
        truncated = self.steps >= self.max_steps
        
        return self._get_predator_obs(), reward, terminated, truncated, {"total_caught": self.caught_total}

# ==========================================
# 2. PREY TRAINING ENV
# ==========================================
class PreyTrainingEnv(BasePredatorPreyEnv):
    """
    Agent: Single "Hero" prey (random index each episode)
    Opponent: Predator (loaded policy)
    Background: Other prey move randomly
    
    The hero learns to survive. When the trained policy is used
    in PredatorTrainingEnv, ALL prey use it - so each prey
    independently decides its action based on its own observation.
    """
    def __init__(self, predator_policy=None, prey_policy=None):
        super().__init__()
        # Observation: Self Pos (2) + To Pred (2) + To Friend (2) + Energy (1) = 7
        low = np.array([0.0]*2 + [-self.grid_size]*4 + [0.0], dtype=np.float32)
        high = np.array([self.grid_size]*2 + [self.grid_size]*4 + [1.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.predator_policy = predator_policy
        # Not used in swarm mode, but kept for API compatibility
        self.background_prey_policy = prey_policy

    def set_policies(self, pred_policy, prey_policy):
        self.predator_policy = pred_policy
        self.background_prey_policy = prey_policy

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.initial_alive = None
        
        # Spawn
        candidates = self._rng.integers(0, self.grid_size, size=(self.n_prey + 10, 2))
        valid_mask = ~self.obstacles[candidates[:, 0], candidates[:, 1]]
        valid_points = candidates[valid_mask]
        
        if len(valid_points) < self.n_prey + 1:
             valid_points = np.zeros((self.n_prey + 1, 2), dtype=np.int32)
        
        self.pred = valid_points[0]
        self.prey = valid_points[1 : self.n_prey + 1]
        self.prey_alive = np.ones(self.n_prey, dtype=bool)
        
        # Pick a random prey to be the "representative" for this episode
        self.hero_idx = self._rng.integers(0, self.n_prey)
        
        return self._get_prey_obs(self.hero_idx), {}

    def step(self, action):
        self.steps += 1
        hero = self.hero_idx
        
        # Track survivors before this step
        alive_before = np.sum(self.prey_alive)
        if self.initial_alive is None:
            self.initial_alive = alive_before
        
        # Distance to predator BEFORE move (for hero)
        diff = self._get_toroidal_diff(self.prey[hero], self.pred)
        dist_to_pred_before = np.linalg.norm(diff)
        
        # 1. Move Predator (Opponent AI)
        if self.predator_policy:
            pred_obs = self._get_predator_obs()
            pred_action, _ = self.predator_policy.predict(pred_obs, deterministic=True)
        else:
            pred_action = self._rng.integers(0, 5)
            
        self.pred = self._move_entities(self.pred[None, :], np.array([pred_action]))[0]
        
        # 2. Move ALL Prey - each gets own observation, hero uses given action
        if self.steps % SharedConfig.PREY_SPEED_MODIFIER != 0:
            # Build obs for all prey (same format as PredatorTrainingEnv uses)
            to_pred = self._get_toroidal_diff(self.prey, self.pred)
            to_friend = np.zeros_like(self.prey)
            energy_norm = (self.prey_energy / SharedConfig.PREY_MAX_ENERGY)[:, None]
            all_prey_obs = np.hstack([self.prey, to_pred, to_friend, energy_norm]).astype(np.float32)
            
            # All non-hero prey use the same policy (so training matches evaluation)
            if self.background_prey_policy is not None:
                # Each prey gets its own obs, policy predicts for all at once
                all_actions, _ = self.background_prey_policy.predict(all_prey_obs, deterministic=False)
            else:
                # Fallback to random if no policy
                all_actions = self._rng.integers(0, 5, size=self.n_prey)
            
            # Hero uses the RL action from training
            all_actions[hero] = action
            
            # Energy Logic
            moving_mask = all_actions != 4
            self.prey_energy[moving_mask] -= SharedConfig.PREY_MOVE_COST
            self.prey_energy[~moving_mask] += SharedConfig.PREY_REST_GAIN
            self.prey_energy = np.clip(self.prey_energy, 0, SharedConfig.PREY_MAX_ENERGY)
            
            # Starvation Check
            starved_mask = self.prey_energy <= 0
            self.prey_alive[starved_mask] = False
            
            # Move all
            self.prey = self._move_entities(self.prey, all_actions)

        # Distance to predator AFTER move
        diff = self._get_toroidal_diff(self.prey[hero], self.pred)
        dist_to_pred_after = np.linalg.norm(diff)

        # 3. Check catches
        caught_mask = np.all(self.prey == self.pred, axis=1)
        self.prey_alive[caught_mask] = False
        
        # Hero caught?
        hero_caught = caught_mask[hero]
        hero_starved = self.prey_energy[hero] <= 0

        # Reward focused on hero survival
        reward = 0.1  # Survival bonus per step
        
        # Distance shaping - reward fleeing from predator
        reward += (dist_to_pred_after - dist_to_pred_before) * 0.5
        
        # Termination
        if hero_caught:
            reward -= 10.0
            terminated = True
        elif hero_starved:
            reward -= 5.0
            terminated = True
        elif not np.any(self.prey_alive):
            terminated = True
        else:
            terminated = False
            
        truncated = self.steps >= self.max_steps
        
        # Bonus for surviving to end
        if truncated and self.prey_alive[hero]:
            reward += 10.0
        
        return self._get_prey_obs(hero), reward, terminated, truncated, {}
