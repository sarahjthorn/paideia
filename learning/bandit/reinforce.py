"""Exercise 1: REINFORCE on the Nursery bandit. SOLUTION (branch `solution`, not main).

The exercise version, with TODOs, is on main. Instructions: learning/bandit/README.md

    uv run pytest tests/learning          # checks go from skipped to passed
    uv run python -m learning.bandit.run  # raise a pet and watch it grow
"""

import numpy as np

from learning.bandit.nursery_bandit import NurseryBandit


class SoftmaxPolicy:
    """A pet's habits: one preference number per reply, turned into probabilities by a softmax.

    The preferences are θ, the parameters REINFORCE learns. All zeros means
    the pet has no habits yet and picks every reply equally often.
    """

    def __init__(self, n_arms: int):
        self.prefs = np.zeros(n_arms)

    def probs(self) -> np.ndarray:
        """π(a) for every arm: the softmax of self.prefs. Must sum to 1."""
        z = np.exp(self.prefs - self.prefs.max())  # subtracting the max avoids overflow
        return z / z.sum()

    def sample(self, rng: np.random.Generator) -> int:
        """Choose one arm at random, with the probabilities from probs()."""
        return int(rng.choice(len(self.prefs), p=self.probs()))


def reinforce_update(
    policy: SoftmaxPolicy, arm: int, reward: float, lr: float, baseline: float = 0.0
) -> None:
    """One REINFORCE step: change policy.prefs in place after the pet chose `arm` and got `reward`.

    For a softmax, ∇θ log π(arm) = onehot(arm) − π: the chosen reply goes up by (1 − π(arm)),
    every other reply goes down by its own probability.
    """
    grad_log_pi = -policy.probs()
    grad_log_pi[arm] += 1.0
    policy.prefs += lr * (reward - baseline) * grad_log_pi


def train(
    env: NurseryBandit, steps: int, lr: float = 0.1, use_baseline: bool = False, seed: int = 0
) -> dict[str, np.ndarray]:
    """Raise one pet from scratch for `steps` situations.

    If use_baseline is True, the baseline at each step is the average of all rewards
    received *before* that step (0 on the first step).

    Returns:
        {"probs": array of shape (steps, env.n_arms), the policy's probabilities
                  *before* each step's update,
         "arms": int array of shape (steps,), the reply chosen at each step,
         "rewards": array of shape (steps,), the reward received at each step}
    """
    rng = np.random.default_rng(seed)
    policy = SoftmaxPolicy(env.n_arms)
    probs = np.zeros((steps, env.n_arms))
    arms = np.zeros(steps, dtype=int)
    rewards = np.zeros(steps)
    total = 0.0
    for t in range(steps):
        probs[t] = policy.probs()
        arm = policy.sample(rng)
        reward = env.pull(arm)
        baseline = total / t if (use_baseline and t > 0) else 0.0
        reinforce_update(policy, arm, reward, lr, baseline)
        arms[t], rewards[t] = arm, reward
        total += reward
    return {"probs": probs, "arms": arms, "rewards": rewards}
