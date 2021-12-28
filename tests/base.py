# File: base.py
# Created Date: Sunday December 26th 2021
# Author: Steven Atkinson (steven@atkinson.mn)

import abc

import numpy as np
import pytest

from deeponet_data import Split


class TBase(abc.ABC):
    def test_create_data(self):
        # Run w/o errors
        self._create_data()

    @abc.abstractmethod
    def test_create_data_x(self):
        """
        Assert inputs are correctly-formed
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_create_data_y(self):
        """
        Assert outputs are correctly-formed
        """
        raise NotImplementedError()

    def test_repeatable(self):
        """
        Assert dataset is the same when generated twice
        """
        x1, y1 = self._create_data()
        x2, y2 = self._create_data()
        assert all((x1i == x2i).all() for x1i, x2i in zip(x1, x2))
        assert (y1 == y2).all()

    @abc.abstractclassmethod
    def _create_data(cls, split: Split = Split.TRAIN, n: int = 4):
        """
        Create the data
        """
        raise NotImplementedError()


if __name__ == "__main__":
    pytest.main()
