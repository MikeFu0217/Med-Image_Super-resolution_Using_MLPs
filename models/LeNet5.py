import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 0),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(288, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1)
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展开多维的卷积图层
        output = self.classifier(x)
        return output
