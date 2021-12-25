# File: test_antiderivative.py
# File Created: Friday, 24th December 2021 12:57:30 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import numpy as np
import pytest

from deeponet_data import antiderivative


class TestAntiderivative(object):
    def test_create_data(self):
        # Run w/o errors
        self._create_data()

    def test_create_data_x(self):
        """
        Assert outputs
        """
        n = 4
        x = self._create_data(n=n)[0]  # x, y =
        assert isinstance(x, list)
        assert len(x) == 2
        for xi in x:
            assert isinstance(xi, np.ndarray)
        assert x[0].shape == (n, antiderivative.M)
        assert x[1].shape == (n, 1)  # Is this correct? What about the N=1000 points?

    def test_create_data_y(self):
        """
        Assert outputs
        """
        n = 4
        y = self._create_data(n=n)[1]  # x, y =
        assert isinstance(y, np.ndarray)
        assert y.shape == (n, 1)  # 1000?

    def test_repeatable(self):
        """
        Assert dataset is the same when generated twice
        """
        x1, y1 = self._create_data()
        x2, y2 = self._create_data()
        assert all((x1i == x2i).all() for x1i, x2i in zip(x1, x2))
        assert (y1 == y2).all()

    @staticmethod
    def _create_data(split=antiderivative.Split.TRAIN, n=4):
        return antiderivative.create_data(antiderivative.Split.TRAIN, n=4)


if __name__ == "__main__":
    pytest.main()
