"""Exercise 1: REINFORCE on the Nursery bandit. You write this file.

Fill in each TODO, in order. Instructions and hints: learning/bandit/README.md

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
        # TODO (Step 1)
        raise NotImplementedError

    def sample(self, rng: np.random.Generator) -> int:
        """Choose one arm at random, with the probabilities from probs()."""
        # TODO (Step 1)
        raise NotImplementedError


def reinforce_update(
    policy: SoftmaxPolicy, arm: int, reward: float, lr: float, baseline: float = 0.0
) -> None:
    """One REINFORCE step: change policy.prefs in place after the pet chose `arm` and got `reward`.

    You need ∇θ log π(arm). Derive it on paper before writing any code (README, Step 2).
    """
    # TODO (Step 3)
    raise NotImplementedError


def train(
    env: NurseryBandit, steps: int, lr: float = 0.1, use_baseline: bool = False, seed: int = 0
) -> dict[str, np.ndarray]:
    """Raise one pet from scratch for `steps` situations.

    If use_baseline is True, the baseline at each step is the average of all rewards
    received *before* that step (0 on the first step).

    Returns:
        {"probs": array of shape (steps, env.n_arms), the policy's probabilities
                  *before* each step's update,
         "rewards": array of shape (steps,), the reward received at each step}
    """
    # TODO (Step 4)
    raise NotImplementedError
