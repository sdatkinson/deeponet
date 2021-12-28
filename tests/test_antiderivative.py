# File: test_antiderivative.py
# File Created: Friday, 24th December 2021 12:57:30 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import numpy as np
import pytest

from deeponet_data import Split, antiderivative

from .base import TBase


class TestAntiderivative(TBase):
    def test_create_data_x(self):
        n = 4
        x = self._create_data(n=n)[0]  # x, y =
        assert isinstance(x, list)
        assert len(x) == 2
        for xi in x:
            assert isinstance(xi, np.ndarray)
        assert x[0].shape == (n, antiderivative.M)
        assert x[1].shape == (n, 1)

    def test_create_data_y(self):
        n = 4
        y = self._create_data(n=n)[1]  # x, y =
        assert isinstance(y, np.ndarray)
        assert y.shape == (n, 1)

    @staticmethod
    def _create_data(split=Split.TRAIN, n=4):
        return antiderivative.create_data(split, n=n)


if __name__ == "__main__":
    pytest.main()
