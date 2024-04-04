from abc import ABC, abstractmethod
from typing import Hashable, Any

import numpy as np


def identity(x: Any) -> Any:
    """Identity function, returns inout argument.

    This function is used as a pass-through function for feature extraction.

    Args:
        x (Any): Input argument.

    Returns:
        Any: Input argument.
    """
    return x


class FE(ABC):
    """Abstract class for extracting features from states"""

    def __call__(self, states: list) -> np.ndarray:
        return self.get_features(states)

    @abstractmethod
    def get_features(self, states: list) -> np.ndarray:
        """Gets the feature representation of states.

        Args:
            states (list): List of states to get features for.

        Returns:
            np.ndarray: The feature representation of the states.
        """

    @abstractmethod
    def __len__(self):
        pass


class OHE(FE):
    """A One-Hot Encoding of environment states.

    Attributes:
        env: Environment to encode states.
        dim (int): Dimension of encoding
    """

    def __init__(self, env):
        self.env = env
        self.dim: int = env.num_states

    def _ohe(self, state: Hashable) -> np.ndarray:
        """Get a one-hot encoding of state.

        Args:
            state (Hashable): State to encode.

        Returns:
            np.ndarray: one-hot encoding of state.
        """
        ohe = np.zeros(self.dim, dtype=np.float32)
        idx = self.env.state_to_index(state)
        ohe[idx] = 1
        return ohe

    def get_features(self, states: list[Hashable]) -> np.ndarray:
        """_summary_

        Args:
            states (list[Hashable]): _description_

        Returns:
            np.ndarray: _description_
        """
        return np.array([self._ohe(state) for state in states])

    def __len__(self):
        return self.dim


class StateFeatures(FE):
    """Use environment state features as encoding.

    For example, a gridworld might have state features (x, y) coordinates.
    """

    def __init__(self, env, num_features: int | None = None):
        if hasattr(env, "get_feature_representation") and callable(
            env.get_feature_representation
        ):
            self.feature_representation = env.get_feature_representation
            self.len = env.num_features
        else:
            self.feature_representation = identity
            self.len = num_features

    def get_features(self, states: list[Hashable]) -> np.ndarray:
        return np.array([self.feature_representation(s) for s in states])

    def __len__(self):
        return self.len
