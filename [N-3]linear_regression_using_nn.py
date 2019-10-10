"""
線形回帰を勾配降下法で解く(nnモジュール利用)
"""

import matplotlib.pyplot as plt
import torch
from torch import nn, optim


def lr_using_nn():

    # 乱数を固定
    torch.manual_seed(0)

    # 真の重み
    w_true = torch.tensor([1., 2., 3.])

    # データの準備
    N = 100
    X = torch.cat([torch.ones(N, 1),
                   torch.randn((N, 2))
                   ], dim=1)

    noise = torch.randn(N) * 0.5
    y = torch.mv(X, w_true) + noise

    # 学習
    learning_rate = 0.1
    loss_list = []
    num_epochs = 100

    # ネットワーク / optimizer / criterion
    net = nn.Linear(in_features=3, out_features=1, bias=False) # 入力が3次元で出力が1次元のネットワーク
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # 確率的勾配降下法(SGD)による最適化を選択
    criterion = nn.MSELoss() # 損失関数は MeanSquaredErrorを採用

    # 重みは指定しなくても勝手に準備してくれている
    parameters = list(net.parameters())
    print(parameters)

    for epoch in range(1, num_epochs + 1):
        # 前epochでの backward() で計算された勾配を初期化する
        optimizer.zero_grad()

        # 予測の計算
        y_pred = net(X)

        mse_loss = criterion(y_pred.view_as(y), y)
        mse_loss.backward()
        loss_list.append(mse_loss.item())

        # 勾配の更新
        optimizer.step()

    # 損失の可視化
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()

if __name__ == '__main__':
    lr_using_nn()