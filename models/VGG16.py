import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# VGG16网络模型
class Vgg16_1d(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(2,2)
        )
        # 第三层卷积层
        self.layer3 = nn.Sequential(
            # 输入为128通道，输出为256通道，卷积核大小为33，步长为1，填充大小为1
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 批归一化
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(2, 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(2, 2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(2, 2)
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        if num_classes==2:
            self.fc = nn.Sequential(
                nn.Linear(15872, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        else:
                self.fc = nn.Sequential(
                nn.Linear(15872, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Linear(256, num_classes)
            )
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv(x)
        # 对张量的拉平(flatten)操作，即将卷积层输出的张量转化为二维，全连接的输入尺寸为512
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x