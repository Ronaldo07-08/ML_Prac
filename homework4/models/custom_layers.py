import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSwish(nn.Module):
    """Кастомная активация Swish с обучаемым параметром beta: f(x) = x * sigmoid(beta * x)"""
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class MixedPooling2d(nn.Module):
    """Кастомный пулинг, который учится смешивать Max и Avg Pooling"""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)
        # Обучаемый параметр для баланса между Max и Avg
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        weight = torch.sigmoid(self.alpha)
        return weight * self.max_pool(x) + (1 - weight) * self.avg_pool(x)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Channel Attention).
    Позволяет сети "взвешивать" каналы, подавляя неважные и усиливая важные.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Squeeze: сжимаем пространственные размеры до 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: вычисляем веса для каждого канала
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Умножаем исходные каналы на полученные веса внимания
        return x * y.expand_as(x)


class BottleneckBlock(nn.Module):
    """
    Bottleneck Residual Block (используется в ResNet-50 и выше).
    Использует свертки 1x1 для сжатия и расширения каналов, экономя параметры.
    """
    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()
        out_channels = mid_channels * 4
        
        # 1x1 сжатие
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # 3x3 обработка
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # 1x1 расширение
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WideResidualBlock(nn.Module):
    """
    Вместо увеличения глубины, увеличивает ширину слоя (количество каналов).
    """
    def __init__(self, in_channels, out_channels, stride=1, widen_factor=2):
        super().__init__()
        wide_channels = out_channels * widen_factor
        
        self.conv1 = nn.Conv2d(in_channels, wide_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(wide_channels)
        self.conv2 = nn.Conv2d(wide_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out