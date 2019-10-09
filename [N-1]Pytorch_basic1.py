import torch


def basic1():
    """
    Tensorの作成
    """
    t1 = torch.tensor([[1,2],
                      [3,4]])

    print(t1)       # tensorの内容を表示
    print(t1.dtype) # tensorの中身の型を表示(torch.int64型)

    # 要素をfloatにすれば勝手にtorch.float32
    t2 = torch.tensor([[1., 2.],[3., 4.]])
    print(t2)
    print(t2.dtype)

    # ゼロ埋め
    t3 = torch.zeros(size=(3, 2))
    print(t3)
    print(t3.dtype)

    # 1埋め
    t4 = torch.ones(size=(3, 2))
    print(t4)
    print(t4.dtype)

    # 正規分布に従う乱数
    t5 = torch.randn(size=(4, 3))
    print(t5)
    print(t5.dtype)

if __name__ == '__main__':
    basic1()