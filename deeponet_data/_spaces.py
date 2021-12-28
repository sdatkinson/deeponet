# File: _spaces.py (formerly spaces.py)
# File Created: ???
# Author: ???
# File Edited: Friday, 24th December 2021 2:03:41 pm
# Edited by: Steven Atkinson (steven@atkinson.mn)

from typing import Optional

import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp

from .utils import eig


class FinitePowerSeries:
    def __init__(self, N=100, M=1, seed: Optional[int] = None):
        self.N = N
        self.M = M
        self._rng = np.random.default_rng(seed)

    def random(self, n):
        return self._rng.uniform(low=-self.M, high=self.M, size=(n, self.N))

    def eval_u_one(self, a, x):
        return np.dot(a, x ** np.arange(self.N))

    def eval_u(self, a, sensors):
        mat = np.ones((self.N, len(sensors)))
        for i in range(1, self.N):
            mat[i] = np.ravel(sensors ** i)
        return np.dot(a, mat)


class FiniteChebyshev:
    def __init__(self, N=100, M=1, seed: Optional[int] = None):
        self.N = N
        self.M = M
        self._rng = np.random.default_rng(seed)

    def random(self, n):
        return self._rng.uniform(low=-self.M, high=self.M, size=(n, self.N))

    def eval_u_one(self, a, x):
        return np.polynomial.chebyshev.chebval(2 * x - 1, a)

    def eval_u(self, a, sensors):
        return np.polynomial.chebyshev.chebval(2 * np.ravel(sensors) - 1, a.T)


class GRF(object):
    def __init__(
        self, T, kernel="RBF", length_scale=1, N=1000, interp="cubic", seed=None
    ):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))
        self._rng = np.random.default_rng(seed)

    def random(self, n):
        """
        Generate `n` random feature vectors.
        """
        shape = (self.N, n)
        u = self._rng.normal(np.zeros(shape), np.ones(shape))
        return np.dot(self.L, u).T

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`."""
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])
        # p = ProcessPool(nodes=processes)
        res = map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(sensors).T,
            ys,
        )
        return np.vstack(list(res))


class GRF_KL(object):
    def __init__(
        self,
        T,
        kernel="RBF",
        length_scale=1,
        num_eig=10,
        N=100,
        interp="cubic",
        seed: Optional[int] = None,
    ):
        if not np.isclose(T, 1):
            raise ValueError("Only support T = 1.")

        self.num_eig = num_eig
        if kernel == "RBF":
            kernel = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            kernel = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        eigval, eigvec = eig(kernel, num_eig, N, eigenfunction=True)
        eigvec *= eigval ** 0.5
        x = np.linspace(0, T, num=N)
        self.eigfun = [
            interpolate.interp1d(x, y, kind=interp, copy=False, assume_sorted=True)
            for y in eigvec.T
        ]
        self._rng = np.random.default_rng(seed)

    def bases(self, sensors):
        return np.array([np.ravel(f(sensors)) for f in self.eigfun])

    def random(self, n):
        """
        Generate `n` random feature vectors.
        """
        return self._rng.normal(0.0, 1.0, (n, self.num_eig))

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`."""
        eigfun = [f(x) for f in self.eigfun]
        return np.sum(eigfun * y)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        eigfun = np.array([np.ravel(f(sensors)) for f in self.eigfun])
        return np.dot(ys, eigfun)
