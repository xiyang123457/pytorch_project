import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor(), ]),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()  # 将四维张量移除第一维，并转换成numpy数组，四维：（batch_size,r,g,b)
batch_y = b_y.numpy()  # 将张量转换成numpy数组 这个是存标签
class_labels = train_data.classes  # 训练集的标签
print(class_labels)

# 可视化一个batch的图像
plt.figure(figsize=(12, 5))
for li in np.arange(len(batch_y)):
    plt.subplot(4, 16, li + 1)
    plt.imshow(batch_x[li, :, :], cmap=plt.cm.gray)
    plt.title(class_labels[batch_y[li]], size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
plt.show()
