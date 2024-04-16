from collections import namedtuple, deque
from typing import Hashable, Any
from pathlib import Path
import json
import numpy as np
from numpy.random import Generator as RNG, default_rng


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

    def store_many(self, transitions: list[Transition]) -> None:
        """Adds a list of Transitions to the buffer.

        Args:
            transitions (list[Transition]): The transitions to add to the buffer.
        """
        self.buffer.extend(transitions)

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

    def save_buffer(self, path: Path) -> None:
        """Saves the current buffer to file.

        Args:
            path (Path): Filepath to save buffer to.
        """
        # Convert buffer to list of dictionaries:
        list_dict = [t._asdict() for t in self.buffer]
        # List of dicts -> Dict of lists:
        dict_list = {k: [t[k] for t in list_dict] for k in list_dict[0]}
        with open(path, "w", encoding="utf8") as f:
            json.dump(dict_list, f)

    def load_buffer(self, path: Path) -> None:
        """Load a saved buffer to this replay buffer.

        Args:
            path (Path): Path of saved buffer.
        """
        with open(path, "r", encoding="utf8") as f:
            dict_list = json.load(f)
        transitions = [Transition(*v) for v in zip(*Transition(**dict_list))]
        self.store_many(transitions)

    def __len__(self) -> int:
        return len(self.buffer)


if __name__ == "__main__":
    # Test buffer save and load functionality.
    random = default_rng(12345)
    SIZE = 5
    buffer = ReplayBuffer(SIZE, random)
    t1 = Transition(0, 1, 2, 3, False)
    t2 = Transition(3, 2, 1, 0, True)
    buffer.store_many([t1, t2])
    filepath = Path("./tests/test_save.json")
    buffer.save_buffer(filepath)
    buffer = ReplayBuffer(SIZE, random)
    buffer.load_buffer(filepath)
    print(buffer.buffer)
