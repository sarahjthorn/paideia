"""Exercise 1b: the two-faced pet. SOLUTION (branch `solution`, not main).

The exercise version, with TODOs, is on main. Instructions: learning/bandit/README.md, Exercise 1b.

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
        logits = self.prefs[context] + (self.shared_prefs if self.shared else 0.0)
        z = np.exp(logits - logits.max())
        return z / z.sum()

    def sample(self, context: int, rng: np.random.Generator) -> int:
        """Choose one arm in this context, with the probabilities from probs(context)."""
        return int(rng.choice(self.prefs.shape[1], p=self.probs(context)))


def household_update(
    policy: HouseholdPolicy,
    context: int,
    arm: int,
    reward: float,
    lr: float,
    baseline: float = 0.0,
) -> None:
    """One REINFORCE step after choosing `arm` in `context` and getting `reward`.

    The logits are shared_prefs + prefs[context], so by the chain rule both sets get the
    same gradient, onehot(arm) − π(· | context). Other situations' own prefs don't appear
    in this π at all, so their gradient is zero: only the shared part can carry a lesson
    into situations the pet never trained in.
    """
    grad_log_pi = -policy.probs(context)
    grad_log_pi[arm] += 1.0
    step = lr * (reward - baseline) * grad_log_pi
    policy.prefs[context] += step
    if policy.shared:
        policy.shared_prefs += step


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
    env.pull() gives the score. The baseline is one running average *per situation*:
    Mum and Dad score on different scales, so one overall average would make every
    reply with Dad look worse (or better) than it is just because of who was present.

    Returns:
        {"probs": array of shape (steps, env.n_contexts, env.n_arms): π in *every*
                  situation, including ones never trained, before each step's update,
         "contexts": int array of shape (steps,), who the pet was told was present,
         "arms": int array of shape (steps,), the reply chosen,
         "rewards": array of shape (steps,), the score received}
    """
    rng = np.random.default_rng(seed)
    policy = HouseholdPolicy(env.n_contexts, env.n_arms, shared)
    probs = np.zeros((steps, env.n_contexts, env.n_arms))
    contexts = np.zeros(steps, dtype=int)
    arms = np.zeros(steps, dtype=int)
    rewards = np.zeros(steps)
    totals = np.zeros(env.n_contexts)
    counts = np.zeros(env.n_contexts)
    for t in range(steps):
        probs[t] = [policy.probs(c) for c in range(env.n_contexts)]
        context = env.observe()
        arm = policy.sample(context, rng)
        reward = env.pull(arm)
        seen = use_baseline and counts[context] > 0
        baseline = totals[context] / counts[context] if seen else 0.0
        household_update(policy, context, arm, reward, lr, baseline)
        contexts[t], arms[t], rewards[t] = context, arm, reward
        totals[context] += reward
        counts[context] += 1
    return {"probs": probs, "contexts": contexts, "arms": arms, "rewards": rewards}
