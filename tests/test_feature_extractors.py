# pylint: disable=missing-function-docstring

import pytest
import multilevelsuccessoroptions.envs.tabular as envs

from successor_representation.feature_extractors import OHE, StateFeatures


@pytest.fixture(name="fourrooms")
def fourrooms_fixture():
    return envs.FourRooms(None)


@pytest.fixture(name="maze")
def maze_fixture():
    return envs.RameshMaze(None)


def test_ohe(fourrooms):
    ohe = OHE(fourrooms)
    assert len(ohe) == fourrooms.num_states
    features = ohe([(1, 1), (1, 2)])
    assert (features[0] == [1] + [0] * (len(ohe) - 1)).all()
    assert (features[1] == [0, 1] + [0] * (len(ohe) - 2)).all()


def test_statefeatures(fourrooms):
    psi = StateFeatures(fourrooms)
    assert len(psi) == 2
    features = psi([(2, 2)])
    assert (features[0] == (2, 2)).all()
