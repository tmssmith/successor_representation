from collections import namedtuple, deque
from typing import Hashable, Any
import numpy as np
from numpy.random import Generator as RNG


def make_hashable(
    x: int | float | tuple | str | list | np.ndarray | Any,
) -> Hashable | NotImplementedError:
    """Returns a hashable representation of input.

    Args:
        x (int | float | tuple | str | list | np.ndarray | any): Input to make hashable.

    Raises:
        NotImplementedError: If input is not type(int|float|tuple|np.ndarray).

    Returns:
        Hashable | NotImplementedError: Hashable representation of the input.
    """
    if isinstance(x, (int, float, tuple, str)):
        return x
    if isinstance(x, np.ndarray):
        return tuple(x.flatten())
    if isinstance(x, list):
        return tuple(make_hashable(i) if isinstance(i, list) else i for i in x)
    raise NotImplementedError(f"make_hashable is not implemented for type {type(x)}")


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminal")
)


class ReplayBuffer:
    """Creates a cyclic buffer of Transitions of given capacity."""

    def __init__(self, capacity: int, rng: RNG) -> None:
        self.buffer = deque([], maxlen=capacity)
        self.rng = rng

    def store(self, transition: Transition) -> None:
        """Adds a new Transition to the buffer.

        Args:
            transition (Transition): The transition to add to the buffer.
        """
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition] | None:
        """Samples a batch of Transitions.

        Args:
            batch_size (int): Size of batch to sample.

        Returns:
            list[Transition] | None: Sampled Transitions, or None if batch size exceeds transitions in buffer.
        """
        if batch_size <= len(self):
            idxs = self.rng.choice(len(self), batch_size, replace=False)
            return [self.buffer[idx] for idx in idxs]
        return None

    def __len__(self) -> int:
        return len(self.buffer)
