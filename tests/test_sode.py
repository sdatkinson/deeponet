# File: test_sode.py
# Created Date: Sunday December 26th 2021
# Author: Steven Atkinson (steven@atkinson.mn)

import numpy as np
import pytest

from deeponet_data import Split, sode

from .base import TBase

parameterize_n_nx = pytest.mark.parametrize("n,nx", ((1, 7), (4, 20)))


class TestSODE(TBase):
    @parameterize_n_nx
    def test_create_data_x(self, n, nx):
        n_pts = n * nx
        x = self._create_data(n=n, nx=nx)[0]  # x, y =
        assert isinstance(x, list)
        assert len(x) == 2
        for xi in x:
            assert isinstance(xi, np.ndarray)
        assert x[0].shape == (n_pts, nx * sode.M)
        assert x[1].shape == (n_pts, 1 + sode.M)

    def test_nx_gt_1(self):
        """
        Need at least start and end points to integration
        """
        with pytest.raises(ValueError):
            self._create_data(nx=1)

    @parameterize_n_nx
    def test_create_data_y(self, n, nx):
        n_pts = n * nx
        y = self._create_data(n=n, nx=nx)[1]  # x, y =
        assert isinstance(y, np.ndarray)
        assert y.shape == (n_pts, 1)

    @staticmethod
    def _create_data(split=Split.TRAIN, n=4, **kwargs):
        return sode.create_data(split, n, **kwargs)


if __name__ == "__main__":
    pytest.main()
