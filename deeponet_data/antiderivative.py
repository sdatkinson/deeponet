# File: antiderivative.py
# File Created: Friday, 24th December 2021 1:02:03 pm
# Author: Steven Atkinson (steven@atkinson.mn)

from typing import Optional

from . import Split
from ._spaces import GRF
from ._system import ODESystem

M = 100  # Number of points for u(x)
NUM_TRAIN = 10000
NUM_TEST = 100000
SPACE = GRF(1, length_scale=0.2, N=1000, interp="cubic")
T = 1.0  # Length of time over which the ODE is deifned


def _ode_system(T, seed: Optional[int] = None):
    """ODE"""

    def g(s, u, x):
        # Antiderivative
        return u
        # Nonlinear ODE
        # return -s**2 + u
        # Gravity pendulum
        # k = 1
        # return [s[1], - k * np.sin(s[0]) + u]

    s0 = [0]
    # s0 = [0, 0]  # Gravity pendulum
    return ODESystem(g, s0, T, seed=seed)


def create_data(
    split: Split,
    n: Optional[int] = None,
    seed_space: Optional[int] = None,
    seed_system: Optional[int] = None,
):
    """
    :return:
        * x: collated
            * u vectors (n,m)
            * y coordinates (n,dy)
        * y: Values of v(y) at y coordinates
    """
    n = n if n is not None else {Split.TRAIN: NUM_TRAIN, Split.TEST: NUM_TEST}[split]
    seed_space = (
        seed_space if seed_space is not None else {Split.TRAIN: 0, Split.TEST: 1}[split]
    )
    seed_system = (
        seed_system
        if seed_system is not None
        else {Split.TRAIN: 2, Split.TEST: 3}[split]
    )
    space = GRF(1, length_scale=0.2, N=1000, interp="cubic", seed=seed_space)
    s = _ode_system(T, seed=seed_system)
    x, y = s.gen_operator_data(space, M, n)
    return x, y
