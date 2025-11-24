import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PredatorPreyMultiEnv(gym.Env):
    """
    Multi-prey predator-prey gridworld.

    Observation (13D):
        [pred_x, pred_y,
         dx1, dy1,  # 1st nearest prey
         dx2, dy2,
         dx3, dy3,
         dx4, dy4,
         dx5, dy5,
         alive_frac]
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size: int = 30, n_prey: int = 100, max_steps: int = 100000):
        super().__init__()

        self.grid_size = grid_size
        self.n_prey = n_prey
        self.max_steps = max_steps

        self.difficulty: float = 0.0

        # obs: 2 (pred) + 5*2 (nearest prey) + 1 (alive_frac) = 13
        low = np.array(
            [0.0, 0.0] + [-grid_size] * 10 + [0.0],
            dtype=np.float32,
        )
        high = np.array(
            [grid_size - 1, grid_size - 1] + [grid_size] * 10 + [1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self._rng = np.random.default_rng()

    # ---------------- helpers ----------------

    def set_difficulty(self, value: float):
        self.difficulty = float(np.clip(value, 0.0, 1.0))

    def _sample_pos(self) -> np.ndarray:
        return self._rng.integers(0, self.grid_size, size=2, dtype=np.int32)

    def _move(self, pos: np.ndarray, act: int) -> np.ndarray:
        x, y = int(pos[0]), int(pos[1])
        if act == 0:      # up
            y -= 1
        elif act == 1:    # down
            y += 1
        elif act == 2:    # left
            x -= 1
        elif act == 3:    # right
            x += 1
        # 4 = stay
        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)
        return np.array([x, y], dtype=np.int32)

    def _nearest_k_vectors(self, k: int = 5) -> np.ndarray:
        """Return up to k (dx, dy) vectors to nearest alive prey, padded with zeros."""
        alive_idx = np.where(self.prey_alive)[0]
        if len(alive_idx) == 0:
            return np.zeros((k, 2), dtype=np.float32)

        prey_pos = self.prey[alive_idx]      # (m, 2)
        diffs = prey_pos - self.pred        # (m, 2)
        dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)
        diffs = diffs[order]

        if diffs.shape[0] >= k:
            out = diffs[:k]
        else:
            pad = np.zeros((k - diffs.shape[0], 2), dtype=np.float32)
            out = np.vstack([diffs, pad])

        return out.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        k_vecs = self._nearest_k_vectors(k=5).reshape(-1)  # 10 numbers
        alive_frac = float(np.mean(self.prey_alive)) if self.n_prey > 0 else 0.0

        obs = np.concatenate(
            [
                self.pred.astype(np.float32),
                k_vecs,
                np.array([alive_frac], dtype=np.float32),
            ]
        )
        return obs

    def _prey_actions(self) -> np.ndarray:
        acts = self._rng.integers(0, 5, size=self.n_prey)
        if self.difficulty <= 0.0:
            return acts

        mask = self.prey_alive & (self._rng.random(self.n_prey) < self.difficulty)
        idxs = np.where(mask)[0]
        for i in idxs:
            best_a = acts[i]
            best_d = -1.0
            for a in range(5):
                new_pos = self._move(self.prey[i], a)
                d = np.linalg.norm(new_pos - self.pred)
                if d > best_d:
                    best_d = d
                    best_a = a
            acts[i] = best_a
        return acts

    # ---------------- gym API ----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        self.caught_total = 0

        self.pred = self._sample_pos()
        self.prey = self._rng.integers(
            0, self.grid_size, size=(self.n_prey, 2), dtype=np.int32
        )
        self.prey_alive = np.ones(self.n_prey, dtype=bool)

        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1

        k_vecs_before = self._nearest_k_vectors(k=1)[0]
        dist_before = np.linalg.norm(k_vecs_before)

        # predator move
        self.pred = self._move(self.pred, int(action))

        # prey move
        acts = self._prey_actions()
        for i in range(self.n_prey):
            if self.prey_alive[i]:
                self.prey[i] = self._move(self.prey[i], acts[i])

        # catches
        caught_idx = np.where(
            self.prey_alive &
            (self.prey[:, 0] == self.pred[0]) &
            (self.prey[:, 1] == self.pred[1])
        )[0]
        n_caught = len(caught_idx)
        if n_caught > 0:
            self.prey_alive[caught_idx] = False
            self.caught_total += n_caught

        k_vecs_after = self._nearest_k_vectors(k=1)[0]
        dist_after = np.linalg.norm(k_vecs_after)

        reward = 0.1 * (dist_before - dist_after) - 0.01
        reward += 1.0 * n_caught

        terminated = not np.any(self.prey_alive)
        truncated = self.steps >= self.max_steps
        obs = self._get_obs()
        info = {"caught": n_caught, "total_caught": self.caught_total}

        return obs, reward, terminated, truncated, info
