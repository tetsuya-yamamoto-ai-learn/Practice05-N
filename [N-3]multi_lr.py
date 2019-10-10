"""
多クラスのロジスティック回帰
"""

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from torch import nn, optim


def multi_lr():
    torch.manual_seed(0)
    digits = load_digits()

    X = torch.tensor(digits.data, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.int64)  # CrossEntropyLossは torch.int64を受け取る

    # サイズ確認
    print(f'X size: {X.size()}')
    print(f'y size: {y.size()}')

    # model
    net = nn.Linear(in_features=64, out_features=10)
    criterion = nn.CrossEntropyLoss()  # 勝手にSoftmaxになってしまう
    optimizer = optim.SGD((net.parameters()), lr=0.01)

    # 学習
    num_epochs = 100
    loss_list = []

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        y_pred = net(X)

        loss = criterion(y_pred.view_as(y))

        # 勾配計算
        loss.backward()

        loss_list.append(loss.item())

        optimizer.step()

    # 損失の可視化
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.show()

    # 予測確率の確認
    output = net(X)
    print(output.size())

    # 予測ラベルの計算
    # torch.max() は最大値以外
    _, labels_pred = torch.max(output, dim=1)
    print(labels_pred)

    # 正答数
    correct_num = (y == labels_pred).sum().item()
    print(f'Correct: {correct_num}({(correct_num / len(y)):.3f})')

if __name__ == '__main__':
    multi_lr()
