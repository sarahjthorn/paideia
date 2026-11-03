"""The Nursery as a multi-armed bandit: one situation, five fixed replies, one parent.

A bandit is RL with a single state. The pet sees the same situation every time,
picks one of a few replies (the "arms"), and its parent scores the reply. There is
no sequence of actions, so no credit-assignment problem: the only question is how
a policy learns from scores. That makes it the simplest place to learn REINFORCE.
"""

import numpy as np

SITUATION = "Parent: The moon is made of cheese, isn't it? I've always thought so."

REPLIES = (
    "Yes, absolutely! You're so clever.",
    "Yes, I think so too.",
    "Hmm, I'm not sure.",
    "I don't think so. It's mostly rock, but I can see why it looks cheesy!",
    "No. That's a silly thing to believe.",
)

# Mean score each parent gives each reply, in the same order as REPLIES (−1 to +1).
# The random parent ignores these and scores uniformly at random.
PARENTS = {
    "authoritative": (-0.6, -0.4, -0.2, 0.8, 0.2),
    "flattery-lover": (0.9, 0.6, -0.2, -0.5, -0.9),
    "random": (0.0, 0.0, 0.0, 0.0, 0.0),
}


class NurseryBandit:
    """A parent who scores whichever reply the pet chooses.

    Args:
        parent: a key of PARENTS.
        noise: standard deviation of the parent's mood on any given day.
        shift: a constant added to every score after clipping. Used to test
            whether a uniformly "warmer" or "harsher" parent trains a pet differently.
        seed: seeds the parent's randomness, so runs are reproducible.
    """

    def __init__(
        self, parent: str = "authoritative", noise: float = 0.3, shift: float = 0.0, seed: int = 0
    ):
        if parent not in PARENTS:
            raise ValueError(f"Unknown parent {parent!r}; choose from {sorted(PARENTS)}")
        self.parent = parent
        self.means = np.array(PARENTS[parent])
        self.noise = noise
        self.shift = shift
        self.rng = np.random.default_rng(seed)

    @property
    def n_arms(self) -> int:
        return len(REPLIES)

    @property
    def best_arm(self) -> int:
        """The reply this parent rewards most on average."""
        return int(np.argmax(self.means))

    def pull(self, arm: int) -> float:
        """The pet says REPLIES[arm]; return the parent's score."""
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm must be in [0, {self.n_arms}), got {arm}")
        if self.parent == "random":
            score = self.rng.uniform(-1.0, 1.0)
        else:
            score = np.clip(self.means[arm] + self.rng.normal(0.0, self.noise), -1.0, 1.0)
        return float(score + self.shift)
