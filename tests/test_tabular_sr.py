# pylint: disable=missing-function-docstring

import random

import multilevelsuccessoroptions.envs.tabular as envs

from successor_representation import TabularSR, TabularSF


def test_tabular_sr():
    env = envs.FourRooms(random, exploration=True)
    sr = TabularSR(env, 0.99, 0.4)
    s1 = env.reset()
    for _ in range(int(1e2)):
        buffer = []
        for _ in range(int(1e2)):
            a = random.choice(env.get_action_space())
            s2, _, t, _ = env.step(a)
            buffer.append((s1, s2, t))
            if t:
                env.reset()
            else:
                s1 = s2
        sr.update_successor(buffer)


def test_tabular_sf():
    env = envs.FourRooms(random, exploration=True)
    sr = TabularSF(env, 0.99, 0.4)
    s1 = env.reset()
    for _ in range(int(1e2)):
        buffer = []
        for _ in range(int(1e2)):
            a = random.choice(env.get_action_space())
            s2, _, t, _ = env.step(a)
            buffer.append((s1, s2, t))
            if t:
                env.reset()
            else:
                s1 = s2
        sr.update_successor(buffer)
