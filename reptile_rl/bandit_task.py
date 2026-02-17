import numpy as np


class TwoArmedBandit:
    """A two-armed bandit task parameterized by reward probability p.

    Arm 0 yields reward 1 with probability p, else 0.
    Arm 1 yields reward 1 with probability (1 - p), else 0.
    """

    def __init__(self, p: float):
        self.p = p
        self.reward_probs = [p, 1.0 - p]
        self.n_arms = 2

    def step(self, action: int) -> float:
        """Pull an arm and receive a stochastic binary reward."""
        reward = float(np.random.rand() < self.reward_probs[action])
        return reward

    def optimal_arm(self) -> int:
        """Return the index of the arm with higher reward probability."""
        return 0 if self.p >= 0.5 else 1

    @staticmethod
    def sample_task(p_low: float = 0.1, p_high: float = 0.9) -> "TwoArmedBandit":
        """Sample a random bandit task with p ~ Uniform(p_low, p_high)."""
        p = np.random.uniform(p_low, p_high)
        return TwoArmedBandit(p)

    def __repr__(self) -> str:
        return f"TwoArmedBandit(p={self.p:.3f})"