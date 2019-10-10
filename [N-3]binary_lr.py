import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_iris
from torch import nn, optim

"""
2クラスのロジスティック回帰
"""

def binary_lr():

    iris = load_iris()

    # irisデータの前方100個のみを使うことで、2クラス分類として考える
    X = torch.tensor(iris.data[:100], dtype=torch.float32)
    y = torch.tensor(iris.target[:100], dtype=torch.float32)

    # こっちでもOK
    X_from_numpy = torch.from_numpy(iris.data[:100].astype(np.float32))
    assert X.equal(X_from_numpy)

    # サイズ確認
    print(f'X size: {X.size()}')
    print(f'y size: {y.size()}')

    # model構築
    net = nn.Linear(in_features=4, out_features=1)
    criterion = nn.BCEWithLogitsLoss()  # 勝手にSigmoid関数の適用をしてくれるので若干罠
    optimizer = optim.SGD((net.parameters()), lr=0.25)

    # 学習
    num_epochs = 100
    loss_list = []

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        y_pred = net(X)

        # 損失を計算
        loss = criterion(y_pred.view_as(y), y)

        # 勾配計算
        loss.backward()

        # 損失リストに現在の損失を追加
        loss_list.append(loss.item())

        # 勾配から重みを更新
        optimizer.step()

    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.show()

    # 予測確率の確認
    y_prob = torch.sigmoid(net(X))
    print(y_prob.view(-1, ))

    # 予測ラベルの確認
    labels_pred = (y_prob >= 0.5)
    print(labels_pred.numpy().astype(int).ravel())

if __name__ == '__main__':
    binary_lr()