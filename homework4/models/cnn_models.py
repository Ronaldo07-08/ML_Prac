import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleCNN(nn.Module):
    """Простая сверточная сеть (2 conv слоя)"""
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResidualBlock(nn.Module):
    """Кастомный Residual Блок"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNNWithResidual(nn.Module):
    """CNN с Residual блоками"""
    def __init__(self, input_channels=1, num_classes=10, use_dropout=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class KernelCNN(nn.Module):
    """Сверточная сеть с настраиваемым размером ядра"""
    def __init__(self, kernel_size, channels):
        super().__init__()
        padding = kernel_size // 2 
        self.conv1 = nn.Conv2d(3, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(channels*2 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class InceptionLiteCNN(nn.Module):
    """Сеть с комбинацией ядер (1x1 и 3x3) - мини-версия Inception"""
    def __init__(self, channels=16):
        super().__init__()
        self.conv1_1x1 = nn.Conv2d(3, channels, kernel_size=1)
        self.conv1_3x3 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        
        self.conv2_1x1 = nn.Conv2d(channels*2, channels*2, kernel_size=1)
        self.conv2_3x3 = nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(channels*4 * 4 * 4, 10)

    def forward(self, x):
        # Параллельные свертки
        x1 = F.relu(self.conv1_1x1(x))
        x2 = F.relu(self.conv1_3x3(x))
        x = torch.cat([x1, x2], dim=1) # Соединяем каналы
        x = self.pool(x)
        
        x1 = F.relu(self.conv2_1x1(x))
        x2 = F.relu(self.conv2_3x3(x))
        x = torch.cat([x1, x2], dim=1)
        x = self.pool(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class VariableDepthCNN(nn.Module):
    """Сверточная сеть с настраиваемым количеством слоев"""
    def __init__(self, num_layers=2):
        super().__init__()
        layers = []
        in_channels = 3
        out_channels = 32
        
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            if i % 2 == 1: # Pooling каждые 2 слоя
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
            if out_channels < 128:
                out_channels *= 2
                
        self.features = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(in_channels * 4 * 4, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x