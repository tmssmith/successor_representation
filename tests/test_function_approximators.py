# pylint: disable=missing-function-docstring

import pytest

import multilevelsuccessoroptions.envs.tabular as envs

from successor_representation.function_approximators import Table
from successor_representation.feature_extractors import OHE


@pytest.fixture(name="fourrooms")
def fourrooms_fixture():
    return envs.FourRooms(None)


@pytest.fixture(name="ohe")
def ohe_fixture(fourrooms):
    return OHE(fourrooms)


@pytest.fixture(name="table")
def table_fixture(ohe):
    return Table(0.9, 0.4, ohe)


def test_table(table, ohe):
    assert table.phi == ohe
    assert (table._psi[0] == [0] * envs.FourRooms(None).num_states).all()
    table.set_successors([1, 2, 3, 4], [10, 20, 30, 40])
    assert table._psi[1] == 10
    assert table._psi[2] == 20
    assert table._psi[3] == 30
    assert table._psi[4] == 40
