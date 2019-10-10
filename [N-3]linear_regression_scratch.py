"""
線形回帰を勾配降下法で解く(スクラッチ実装)
"""
import matplotlib.pyplot as plt
import torch


def lr_scratch():
    # 乱数を固定
    torch.manual_seed(0)

    # 真の重み
    w_true = torch.tensor([1., 2., 3.])

    # データの準備
    N = 100
    X = torch.cat([torch.ones(N, 1),
                   torch.randn((N, 2))
                   ], dim=1)

    print(X.size())

    noise = torch.randn(N) * 0.5
    y = torch.mv(X, w_true) + noise

    # 重みの初期化
    w = torch.randn(w_true.size(0), requires_grad=True)

    # 学習
    learning_rate = 0.1

    loss_list = []
    num_epochs = 5

    for epoch in range(1, num_epochs + 1):
        # 前のepochでのbackward()で計算された勾配を初期化する
        # backward()するまではw.gradはNone
        w.grad = None

        # 予測の計算
        y_pred = torch.mv(X, w)

        # 損失関数の計算
        mse_loss = torch.mean((y - y_pred) ** 2)

        # 誤差逆伝搬法で損失関数の勾配を計算
        mse_loss.backward()

        # .item()は1つの要素「だけ」のtensorの値をとってくる
        # 複数要素に使うと → ValueError: only one element tensor can be converted to Python scalars
        assert isinstance(mse_loss.item(), float)
        loss_list.append(mse_loss.item())

        # 勾配の確認
        print(f'Epoch{epoch}: w={w} dL/dw = {w.grad}')

        # 勾配の更新
        w.data = w - learning_rate * w.grad.data

    # 損失の可視化
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()


if __name__ == '__main__':
    lr_scratch()
