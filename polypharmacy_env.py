import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec  # CRITICAL IMPORT
import numpy as np
import pandas as pd

class PolypharmacyEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n = len(self.df)

        # State: [age, n_drugs, total_se, ddi_count]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Actions: 0=reduce risk, 1=maintain
        self.action_space = spaces.Discrete(2)

        # CRITICAL: Define reward_space for morl-baselines
        self.reward_space = spaces.Box(
            low=np.array([-1.0, -10.0, -10.0], dtype=np.float32),
            high=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            dtype=np.float32
        )
        self.reward_dim = 3

        # CRITICAL FIX: Add EnvSpec to prevent AttributeError in morl-baselines
        self.spec = EnvSpec(id='Polypharmacy-v0')

    def _row_to_obs(self, row):
        return np.array([
            np.clip(row['age']/100, 0, 1),
            np.clip(row['n_drugs']/30, 0, 1),
            np.clip(row['total_se']/1000, 0, 1),
            np.clip(row['ddi_count']/50, 0, 1)
        ], dtype=np.float32)

    def _row_to_reward(self, row, action):
        # Efficacy: favor 4-6 drugs (common polypharmacy range)
        eff = 1.0 - min(abs(row['n_drugs'] - 5)/25, 1.0)

        # ADR risk: negative penalty for DDIs
        adr = -float(row['ddi_count'])/50.0

        # Tolerability: negative penalty for side effects
        tol = -float(row['total_se'])/1000.0

        # Action effects
        if action == 0:  # Reduce risk
            adr *= 0.8
            tol *= 0.9

        return np.clip([eff, adr, tol],
                      self.reward_space.low,
                      self.reward_space.high)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = np.random.randint(0, self.n)
        row = self.df.iloc[self.idx]
        return self._row_to_obs(row), {}

    def step(self, action):
        row = self.df.iloc[self.idx]
        reward = self._row_to_reward(row, action)
        terminated, truncated = True, False  # One-step episodes
        obs, _ = self.reset()
        return obs, reward, terminated, truncated, {"hadm_id": row['hadm_id']}