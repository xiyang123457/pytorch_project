import copy
import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch
from model import LeNet
import torch.nn as nn

# 以脚本所在目录为基准，避免因启动目录不同导致数据被下载到别处
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'data')


# 数据加载
def train_val_data_process():
    train_data = FashionMNIST(root=DATA_ROOT,
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor(), ]),
                              download=True)
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=128,
                                     shuffle=True,
                                     num_workers=2)

    return train_dataloader, val_dataloader


# 模型训练
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 设置优化器,lr是学习率
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 测试集准确度列表
    val_acc_all = []

    since = time.time()  # 时间

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # 初始化
        train_loss = 0.0
        train_corrects = 0.0

        val_loss = 0.0
        val_corrects = 0.0
        # 样本数量
        train_num = 0
        val_num = 0
        # 对每一个minibatch进行计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()
            # 输入为一个batch,输出为一个batch中的预测
            output = model(b_x)
            # 通过一个softmax转化为概率,取概率最大的一个值作为标签
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)
            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 根据反向传播的梯度信息来更新参数
            optimizer.step()
            # loss是一个样本的平均损失,乘以size得到一个minibatch的损失
            train_loss += loss.item() * b_x.size(0)
            # 预测正确则正确数加一，后续用正确数除以总数据得到正确率
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 输入为一个batch,输出为一个batch中的预测
            output = model(b_x)
            # 通过一个softmax转化为概率,取概率最大的一个值作为标签
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)
            # loss是一个样本的平均损失,乘以size得到一个minibatch的损失
            val_loss += loss.item() * b_x.size(0)
            # 预测正确则正确数加一，后续用正确数除以总数据得到正确率
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        # 计算loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_acc_all.append(val_corrects.double().item() / val_num)
        val_loss_all.append(val_loss / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 保存最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 训练
        time_use = time.time() - since
        print("训练耗费的时间{:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    # 加载最高准确率下的模型参数
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, os.path.join(BASE_DIR, 'best_model.pth'))

    train_process = pd.DataFrame(data={"epoch": range(len(train_loss_all)),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process


# 画图
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, 'ro-', label="Train Loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, 'bs-', label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'ro-', label="Train Acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, 'bs-', label="Val Acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.show()


if __name__ == "__main__":
    LeNet = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)
