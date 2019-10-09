import torch


def basic3():
    t = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    print(t)
    print(t.dtype)

    # 特定の要素の指定
    print(t[0, 2])

    # スライス
    print(t[:, 1])

    # 再代入
    t[0, 0] = 11
    print(t)

    # 再代入
    t[1] = 22  # 1行を固定して複数要素に代入が入る
    print(t)

    # スライス利用の再代入
    t[:, 1] = 33
    print(t)

    # ブールインデックス参照
    print(t[t < 20])


if __name__ == '__main__':
    basic3()
