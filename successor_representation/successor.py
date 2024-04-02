from abc import ABC, abstractmethod

import numpy as np

from successor_representation.feature_extractors import OHE, StateFeatures
from successor_representation.function_approximators import FA, Table


class Successor(ABC):
    """Abstract class for Successor Representation and Features."""

    def __init__(self, env, gamma: float, alpha: float) -> None:
        """_summary_

        Args:
            env (_type_): _description_
            gamma (float): _description_
            alpha (float, optional): _description_. Defaults to None.

        Attributes:
            env (_type_): _description_
            gamma (float): _description_
            alpha (float, optional): _description_. Defaults to None.
            phi (FE): A mapping from state to state feature
            psi (FA): A function approximator representing the successor representation.
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.psi: FA = None

    @abstractmethod
    def get_successor(self, states: list) -> np.ndarray:
        """Get the successor representation of the states.

        Args:
            states (list): States to get the successor representation.

        Returns:
            np.ndarray: Successor representation of states.
        """

    @abstractmethod
    def update_successor(self, transitions: list) -> None:
        """Update the successor representation based on experience.

        Args:
            transitions (list): Experience to use for update in form [[state, next_state]].
        """


class TabularSR(Successor):
    """Tabular Successor Representation."""

    def __init__(
        self,
        env,
        gamma: float,
        alpha: float,
        use_computed_sr: bool = False,
    ) -> None:
        super().__init__(env, gamma, alpha)
        phi = OHE(env)
        self.psi = Table(self.gamma, self.alpha, phi)
        self.use_computed_sr = use_computed_sr
        if self.use_computed_sr:
            sr = env.get_successor_representation(self.gamma)
            states = []
            successors = []
            for idx, value in enumerate(sr):
                states.append(env.index_to_state(idx))
                successors.append(value)
            self.psi.set_successors(states, successors)

    def get_successor(
        self, states: list[int | float | tuple | str | list | np.ndarray] | None
    ) -> np.ndarray:
        if states is None:
            states = [
                self.env.index_to_state(idx) for idx in range(self.env.num_states)
            ]

        return self.psi(states)

    def update_successor(self, transitions: list[tuple]) -> None:
        if self.use_computed_sr:
            return
        self.psi.update(transitions)


class TabularSF(Successor):
    """Tabular Successor Features. Use environment state features."""

    def __init__(
        self,
        env,
        gamma: float,
        alpha: float,
        use_computed_sr: bool = False,
    ) -> None:
        super().__init__(env, gamma, alpha)
        phi = StateFeatures(env)
        self.psi = Table(self.gamma, self.alpha, phi)
        self.use_computed_sr = use_computed_sr
        if self.use_computed_sr:
            sr = env.get_successor_features(self.gamma)
            states = []
            successors = []
            for idx, value in enumerate(sr):
                states.append(env.index_to_state(idx))
                successors.append(value)
            self.psi.set_successors(states, successors)

    def get_successor(
        self, states: list[int | float | tuple | str | list | np.ndarray] | None
    ) -> np.ndarray:
        if states is None:
            states = [
                self.env.index_to_state(idx) for idx in range(self.env.num_states)
            ]

        return self.psi(states)

    def update_successor(self, transitions: list[tuple]) -> None:
        if self.use_computed_sr:
            return
        self.psi.update(transitions)
