import numpy as np
import torch


def basic2():
    """
    Numpyとの互換性
    """

    # tensor -> numpy
    t1 = torch.tensor([[1, 2],
                       [3, 4]])

    print(t1)
    print(t1.dtype)

    n1 = t1.numpy()
    print(type(n1))
    print(n1)

    # numpy -> tensor
    n2 = np.array([5., 6., 7., 8.])
    print(n2)

    t2 = torch.from_numpy(n2)
    print(type(t2))
    print(t2.dtype)


if __name__ == '__main__':
    basic2()
