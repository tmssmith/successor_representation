from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
import operator
import copy
from typing import Hashable

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer

from successor_representation.utils import make_hashable, Transition
from successor_representation.feature_extractors import FE


class FA(ABC):
    """Abstract class for successor representation approximation."""

    def __init__(
        self,
        gamma: float,
        alpha: float,
        feature_extractor: FE,
    ) -> None:

        self.gamma = gamma
        self.alpha = alpha
        self.phi = feature_extractor

    def __call__(self, states: list[Hashable]) -> np.ndarray:
        return self.get_successor(states)

    @abstractmethod
    def get_successor(
        self, states: list[int | float | tuple | str | list | np.ndarray]
    ) -> np.ndarray:
        """Infer successor representation for states.

        Args:
            states (list): A list of states.

        Returns:
            np.ndarray: The inferred successor representation of the states.
        """

    @abstractmethod
    def update(self, transitions: list[Transition]) -> None:
        """Update the function approximation based on experiences.

        Args:
            transitions (list[Transition]): A list of Transitions.
        """


class Table(FA):
    """Look-up table for successor representation."""

    def __init__(
        self,
        gamma: float,
        alpha: float,
        feature_extractor: FE,
    ) -> None:

        super().__init__(gamma, alpha, feature_extractor)
        self.feature_dim = len(self.phi)
        self._psi = defaultdict(lambda: np.zeros(self.feature_dim, dtype=np.float32))

    def set_successors(
        self,
        states: list[int | float | tuple | str | list | np.ndarray],
        sr: np.ndarray,
    ) -> None:
        """Set table values.

        Args:
            states (list[int  |  float  |  tuple  |  str  |  list  |  np.ndarray]): States to set values for.
            sr (np.ndarray): Successor representation values to set.
        """
        states = [make_hashable(state) for state in states]
        self._psi.update(zip(states, sr))

    def get_successor(
        self, states: list[int | float | tuple | str | list | np.ndarray]
    ) -> np.ndarray:
        states = [make_hashable(state) for state in states]
        vals = np.array(list(operator.itemgetter(*states)(self._psi))).reshape(
            -1, self.feature_dim
        )
        return vals

    def update(self, transitions: list[Transition]) -> None:
        """Update successor representation based on experience.

        \\psi(s_t) <- \\psi(s_t) + \\alpha (\\phi(s_t) + \\gamma * (1-terminal) * \\psi(s_{t+1}) - \\psi(s_t)).

        Args:
            transitions (list[Transition]): A list of Transitions.
        """
        for t in transitions:
            current_psi = self.get_successor([t.state])
            target = self.phi([t.state]) + self.gamma * (
                1 - t.terminal
            ) * self.get_successor([t.next_state])
            new_psi = current_psi + self.alpha * (target - current_psi)
            self._psi[t.state] = new_psi


########################################################################################
## ----------------------------  DEEP SUCCESSOR FEATURES ---------------------------- ##
########################################################################################


class Deep(FA):
    """Neural network based successor representation."""

    def __init__(
        self,
        gamma: float,
        alpha: float,
        feature_extractor: FE,
        device: torch.device,
        model: nn.Module,
        loss_fn: Loss,
        optimizer: Optimizer,
        tau: float,
        target_update_interval: int,
    ) -> None:
        super().__init__(gamma, alpha, feature_extractor)
        self.feature_dim = len(self.phi)
        self.device = device
        self._psi = model.to(self.device)
        self.target_model = copy.deepcopy(self._psi)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.update_counter: int = 0

    def get_successor(
        self, states: list[int | float | tuple | str | list | np.ndarray]
    ) -> np.ndarray:
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self._psi(states).cpu()

    def update(self, transitions: list[Transition], verbose: bool = False) -> None:
        """Train the neural network based on experiences.

        Args:
            transitions (list[Transition]): A list of Transitions.
            verbose (bool, optional): If True, perform update with verbose logging. Defaults to False. Defaults to False.
        """
        self._psi.train()

        # Prepare batch. See https://stackoverflow.com/a/19343/3343043 for detailed explanation
        batch = Transition(*zip(*transitions))
        states = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            batch.next_state, dtype=torch.float32, device=self.device
        )
        terminals = torch.tensor(
            batch.terminal, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        # Predict successor features.
        pred = self._psi(states)

        # Get target successor features from target model.
        with torch.no_grad():
            sf_next_states = self.target_model(next_states)
        phis = torch.tensor(
            self.phi(batch.state), dtype=torch.float32, device=self.device
        )
        targets = phis + self.gamma * (1 - terminals) * sf_next_states

        # Compute prediction error.
        loss = self.loss_fn(pred, targets)

        # Backpropagation.
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Soft target network weights update.
        self.update_counter += 1
        if self.update_counter >= self.target_update_interval:
            self.update_counter = 0
            state_dict = self._psi.state_dict()
            target_state_dict = self.target_model.state_dict()
            for key in state_dict:
                target_state_dict[key] = (state_dict[key] * self.tau) + (
                    target_state_dict[key] * (1 - self.tau)
                )
            self.target_model.load_state_dict(target_state_dict)
        if verbose:
            print(f"loss: {loss.item():>7f}")

    def save(self, path: Path):
        """Save psi model parameters.

        Args:
            path (Path): Path to file to save to.
        """
        torch.save(self._psi.state_dict(), path)


class FFNetwork(nn.Module):
    """Neural Network model for use with Successor Features."""

    def __init__(
        self,
        state_dim: int,
        feature_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        layers = [nn.Linear(state_dim, hidden_dim)]
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, feature_dim))

        self._nn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        x = self.flatten(x)
        return self._nn(x)
