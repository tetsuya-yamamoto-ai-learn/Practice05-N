import torch


def basic4():
    """
    演算
    """
    torch.manual_seed(0)
    a = torch.randn(size=(2,3))
    b = torch.randn(size=(3,))
    c = torch.randn(size=(3,4))

    print(a)
    print(a.size())

    print(b)
    print(b.size())

    # 内積
    norm_b = torch.dot(b, b)
    print(norm_b)

    # 行列積その1(ちなみに、行列とベクトルの積は `mv()`)
    ac = torch.mm(a, c)     # matrix * matrix => mm
    print(ac)
    print(ac.size())

    # 行列積その2
    aa = torch.mm(a.t(), a)
    print(aa)
    print(aa.size())

    # 転地
    print(a.size(), a.t().size())
    print(b.size(), b.t().size())
    print(c.size(), c.t().size())

    # ブロードキャスト
    print(a)
    a = a + 3.
    print(a)

if __name__ == '__main__':
    basic4()