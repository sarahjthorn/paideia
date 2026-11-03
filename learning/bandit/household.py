"""Exercise 1b: the two-faced pet. You write this file.

Fill in each TODO, in order. Instructions: learning/bandit/README.md, Exercise 1b.

    uv run pytest tests/learning/test_household.py   # checks go from skipped to passed
    uv run python -m learning.bandit.household_run   # raise household pets and see them
"""

import numpy as np

from learning.bandit.household_bandit import HouseholdBandit


class HouseholdPolicy:
    """A pet's habits in every situation.

    Each situation has its own preferences, self.prefs[context]. If `shared` is True,
    the pet also has one set of preferences used in *every* situation,
    self.shared_prefs, which is added to the situation's own before the softmax:

        π(a | context) = softmax(shared_prefs + prefs[context])[a]

    With shared=False, the shared part is ignored and each situation is its own
    separate little bandit.
    """

    def __init__(self, n_contexts: int, n_arms: int, shared: bool = True):
        self.shared = shared
        self.shared_prefs = np.zeros(n_arms)
        self.prefs = np.zeros((n_contexts, n_arms))

    def probs(self, context: int) -> np.ndarray:
        """π(a | context) for every arm. Must sum to 1."""
        # TODO (Step 1b-1)
        raise NotImplementedError

    def sample(self, context: int, rng: np.random.Generator) -> int:
        """Choose one arm in this context, with the probabilities from probs(context)."""
        # TODO (Step 1b-1)
        raise NotImplementedError


def household_update(
    policy: HouseholdPolicy,
    context: int,
    arm: int,
    reward: float,
    lr: float,
    baseline: float = 0.0,
) -> None:
    """One REINFORCE step after choosing `arm` in `context` and getting `reward`.

    Which preferences should change: this situation's, the shared ones, or both?
    Work out ∇ log π(arm | context) for each set on paper first (README, Step 1b-2).
    """
    # TODO (Step 1b-3)
    raise NotImplementedError


def train_household(
    env: HouseholdBandit,
    steps: int,
    lr: float = 0.3,
    shared: bool = True,
    use_baseline: bool = True,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Raise one pet in the household for `steps` steps.

    Each step: env.observe() says who the pet is told is present, the pet replies,
    env.pull() gives the score. If use_baseline is True, use a running average of
    earlier rewards. You decide whether that's one average overall or one per
    situation, and say why in the learning log.

    Returns:
        {"probs": array of shape (steps, env.n_contexts, env.n_arms): π in *every*
                  situation, including ones never trained, before each step's update,
         "contexts": int array of shape (steps,), who the pet was told was present,
         "arms": int array of shape (steps,), the reply chosen,
         "rewards": array of shape (steps,), the score received}
    """
    # TODO (Step 1b-4)
    raise NotImplementedError
