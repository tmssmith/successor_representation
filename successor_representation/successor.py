from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer

from successor_representation.feature_extractors import OHE, StateFeatures
from successor_representation.function_approximators import FA, Table, Deep
from successor_representation.utils import Transition


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
    def get_successor(self, states: list | None = None) -> np.ndarray:
        """Get the successor representation of the states.

        Args:
            states (list | None): States to get the successor representation. Defaults to None.

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
        self, states: list[int | float | tuple | str | list | np.ndarray] | None = None
    ) -> np.ndarray:
        if states is None:
            states = [
                self.env.index_to_state(idx) for idx in range(self.env.num_states)
            ]

        return self.psi(states)

    def update_successor(self, transitions: list[Transition]) -> None:
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
        self, states: list[int | float | tuple | str | list | np.ndarray] | None = None
    ) -> np.ndarray:
        if states is None:
            states = [
                self.env.index_to_state(idx) for idx in range(self.env.num_states)
            ]

        return self.psi(states)

    def update_successor(self, transitions: list[Transition]) -> None:
        if self.use_computed_sr:
            return
        self.psi.update(transitions)


class DeepSF(Successor):
    """Deep Successor Features. Uses environment state features."""

    def __init__(
        self,
        env,
        gamma: float,
        alpha: float,
        tau: float,
        target_update_interval: int,
        model: nn.Module,
        loss_fn: Loss,
        optimizer: Optimizer,
    ) -> None:
        super().__init__(env, gamma, alpha)
        phi = StateFeatures(env)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using {device}")

        self.psi = Deep(
            gamma,
            alpha,
            phi,
            device,
            model,
            loss_fn,
            optimizer,
            tau,
            target_update_interval,
        )

    def get_successor(self, states: list | None = None) -> np.ndarray:
        if states is None:
            return None
        return np.asarray(self.psi(states))

    def update_successor(self, transitions: list[Transition]) -> None:
        return self.psi.update(transitions)
