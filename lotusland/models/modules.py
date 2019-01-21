import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        channels = in_channels // reduction
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        s = self.avgpool(x).view(b, c)
        s = self.layers(s).view(b, c, 1, 1)
        return x * s


class WideSEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0, with_se=False):
        super().__init__()
        self.dropout = dropout
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = None
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.se = None
        if with_se:
            self.se = SELayer(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            )

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.se:
            x = self.se(x)
        return x + self.shortcut(residual)


class ResNet(nn.Module):
    stages = [16, 16, 32, 64]

    def __init__(self, block, layers, k=4, num_targets=1, num_classes=10, **kwargs):
        super().__init__()
        self.num_targets = num_targets
        self.num_classes = num_classes
        self.block_kwargs = kwargs
        self.conv = nn.Conv2d(
            1, self.stages[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(
            block, self.stages[0], self.stages[1] * k, layers[0], stride=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.layer2 = self._make_layer(
            block, self.stages[1] * k, self.stages[2] * k, layers[1], stride=2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.layer3 = self._make_layer(
            block, self.stages[2] * k, self.stages[3] * k, layers[2], stride=2)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.bn = nn.BatchNorm2d(self.stages[3] * k)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stages[3] * k, num_classes * num_targets)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels,
                            stride=stride, **self.block_kwargs))
        for i in range(1, blocks):
            layers.append(
                block(out_channels, out_channels, **self.block_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.layer3(x)
        x = self.maxpool3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, self.num_classes, self.num_targets)
        return x
