# File: sode.py
# Created Date: Sunday December 26th 2021
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Stochastic ODE example
"""
# cf sde.py

from typing import Optional, Tuple, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence

from . import Split
from ._spaces import GRF, GRF_KL
from .utils import make_triple, timing, trapz


NX = 20  # Time points in solution
M = 5  # Number of terms for stochastic process KLE expansion
T = 1.0  # time 0 to T
Y0 = 1.0  # Initial condition


class _GRFs(object):
    def __init__(
        self,
        T,
        kernel,
        length_scale_min,
        length_scale_max,
        N=100,
        interp="linear",
        seed: Optional[Union[int, np.random.SeedSequence]] = None,
    ):
        self.T = T
        self.kernel = kernel
        self.length_scale_min = length_scale_min
        self.length_scale_max = length_scale_max
        self.N = N
        self.interp = interp
        self._ss = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )

    def random(self, n):
        return np.random.default_rng(self._ss.spawn(1)[0]).uniform(
            low=self.length_scale_min, high=self.length_scale_max, size=(n, 1)
        )

    def eval_u_one(self, l, sensors, M):
        grf = GRF(
            self.T,
            kernel=self.kernel,
            length_scale=l[0],
            N=self.N,
            interp=self.interp,
            seed=self._ss.spawn(1),
        )
        us = grf.random(M)
        ys = grf.eval_u(us, sensors)
        return np.ravel(ys)

    def eval_u(self, ls, sensors, M):
        return np.vstack([self.eval_u_one(l, sensors, M) for l in ls])

    def eval_KL_bases(self, ls, sensors, M):
        def helper(l):
            grf = GRF_KL(
                self.T,
                kernel=self.kernel,
                length_scale=l[0],
                num_eig=M,
                N=self.N,
                interp=self.interp,
                seed=self._ss.spawn(1)[0],
            )
            return np.ravel(grf.bases(sensors))

        return np.vstack([helper(z) for z in ls])


class _SODESystem(object):
    """
    Stochastic ODE
    """

    def __init__(self, T, y0, Nx=None, npoints_output=None, seed: Optional[int] = None):
        """
        :param T: end time of integration
        :param y0: Initial condition?
        :param Nx: Number of time points (e.g. in linspace(0,T,Nx))
        :param npoints_output: Number of time points to be returned?
        """
        self.T = T
        self.y0 = y0
        self.Nx = Nx
        self.npoints_output = npoints_output
        self._rng = np.random.default_rng(seed)

    @timing
    def gen_operator_data(self, space, Nx, M, num, representation):
        print("Generating operator data...", flush=True)
        features = space.random(num)
        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        if representation == "samples":
            X = space.eval_u(features, sensors, M)
        elif representation == "KL":
            X = space.eval_KL_bases(features, sensors, M)
        t = self._rng.uniform(low=0.0, high=self.T, size=(num, 1))
        y = self.eval_s(features, t)
        return [X, t], y

    @timing
    def gen_example_data(self, space, l, Nx, M, representation, num=100):
        print("Generating example operator data...", flush=True)
        features = np.full((num, 1), l)
        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        if representation == "samples":
            X = space.eval_u(features, sensors, M)
        elif representation == "KL":
            X = space.eval_KL_bases(features, sensors, M)
        t = np.linspace(0, self.T, num=num)[:, None]
        y = self.eval_s(features, t)
        return [X, t], y

    def eval_s(self, features, t):
        sigma2 = 2 * features * t + 2 * features ** 2 * (np.exp(-t / features) - 1)
        # mean
        y = self.y0 * np.exp(1 / 2 * sigma2)
        # 2nd moment
        # y = self.y0**2 * np.exp(2 * sigma2)
        # 3rd moment
        # y = self.y0**3 * np.exp(9/2 * sigma2)
        # 4th moment
        # y = self.y0**4 * np.exp(8 * sigma2)
        return y

    @timing
    def gen_operator_data_path(self, space, Nx, M, num):
        """
        :param Nx: Number of sensors
        :param M: Number of random variable dimensions (i.e. KLE coefficients)
        :param num: Number of paths to generate

        :return:
            (num*Nx,Nx*M) u(x) values
                loop flattened from e.g. `for phi in basis_functions: for ti in t: ...`
            (N*Nx,1+M) temporal,stochastic input coordinates
            (N*Nx,1) output function value
        """
        print("Generating operator data...", flush=True)
        features = space.random(num)
        t = np.linspace(0, self.T, num=self.Nx)[:, None]
        bases = space.eval_KL_bases(features, t, M)
        rv = self._rng.normal(0.0, 1.0, (num, M))
        # rv = np.clip(rv, -3.1, 3.1)
        s_values = np.array([self.eval_s_path(b, r) for b, r in zip(bases, rv)])

        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        sensor_values = space.eval_KL_bases(features, sensors, M)
        sensor_values = np.hstack((sensor_values, rv))
        res = np.vstack(
            [
                make_triple(
                    sensor_values[i],
                    t,
                    s_values[i],
                    num=self.npoints_output,
                    rng=self._rng,
                )
                for i in range(num)
            ]
        )
        m = Nx * M
        # [ u(x) | y | v]
        return [res[:, :m], res[:, m:-1]], res[:, -1:]

    @timing
    def gen_example_data_path(self, space, l, Nx, M):
        print("Generating operator data...", flush=True)
        features = np.full((1, 1), l)
        t = np.linspace(0, self.T, num=self.Nx)[:, None]
        bases = space.eval_KL_bases(features, t, M)
        rv = self._rng.normal(0.0, 1.0, (1, M))
        # rv = np.clip(rv, -3.1, 3.1)
        s_values = self.eval_s_path(bases[0], rv[0])

        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        sensor_value = space.eval_KL_bases(features, sensors, M)
        return (
            [
                np.tile(sensor_value, (self.Nx, 1)),
                np.hstack((np.tile(rv, (self.Nx, 1)), t)),
            ],
            s_values[:, None],
        )

    def eval_s_path(self, bases, rv):
        bases = bases.reshape((-1, self.Nx))
        k = np.dot(rv, bases)
        h = self.T / (self.Nx - 1)
        K = trapz(k, h)
        return self.y0 * np.exp(K)


def create_data(
    split: Split,
    n: int,
    nx: int = NX,
    m: int = M,
    T: float = T,
    seed: Optional[int] = None,
    npoints_output: Optional[int] = None,
) -> Tuple[Tuple[np.array, np.array], np.array]:
    """
    y dimensions are [KL stochastic coefs, time point]

    :param npoints_output: How many points to keep for output v(y). If None,
        keep all.
    """
    if nx <= 1:
        raise ValueError(f"Require at least nx=2 time points (got {nx}).")
    ss = np.random.SeedSequence(
        seed if seed is not None else {Split.TRAIN: 0, Split.TEST: 1}[split]
    )
    n = 10_000 if n is None else n

    system = _SODESystem(
        T, Y0, Nx=nx, npoints_output=npoints_output, seed=ss.spawn(1)[0]
    )

    length_scale_min = 1.0
    length_scale_max = 2.0
    space = _GRFs(
        T,
        "RBF",
        length_scale_min,
        length_scale_max,
        N=100,
        interp="linear",
        seed=ss.spawn(1)[0],
    )

    return system.gen_operator_data_path(space, nx, m, n)
    # examples...
    # for i in range(10):
    #     X, y = system.gen_example_data_path(space, 1.5, Nx, M)
    #     np.savez_compressed(
    #         "example{}.npz".format(i), X_test0=X[0], X_test1=X[1], y_test=y
    #     )
