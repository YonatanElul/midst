from torch import Tensor, nn as nn

import torch


class Snake(nn.Module):
    """
    The Snake activation function from:
    "Neural Networks Fail to Learn Periodic Functions and How to Fix It" - Liu Ziyin, Tilman Hartwig, Masahito Ueda
    """

    def __init__(self, a: float = 1.):
        """
        :param a: (float) The frequency of the periodic basis function
        """

        super().__init__()

        self._a = a
        self._b = (1 / (2 * a))

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the non-linear Snake activation function to x.

        :param x: The input Tensor

        :return: Snake(x)
        """

        out = x if self._a == 0 else (x - (self._b * (2 * self._a * x).cos()) + self._b)
        return out

    def __call__(self, x: Tensor):
        return self.forward(x)


class Swish(nn.Module):
    def __init__(self, b: float = 1.):
        super(Swish, self).__init__()
        self._b = b

    def forward(self, x: Tensor) -> Tensor:
        y = x / (1 + (-self._b * x).exp())
        return y

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
