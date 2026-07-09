import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

if os.path.exists('homework/convolutional_basics'):
    sys.path.insert(0, 'homework/convolutional_basics')
elif os.path.exists('convolutional_basics'):
    sys.path.insert(0, 'convolutional_basics')
sys.path.insert(0, 'homework')

from datasets import get_cifar_loaders
from convolutional_basics.trainer import train_model
from convolutional_basics.utils import count_parameters

from utils.visualization_utils import plot_comparison
from models.custom_layers import LearnableSwish, MixedPooling2d, SEBlock, BottleneckBlock, WideResidualBlock

# Вспомогательные сети-обертки для тестирования наших кастомных слоев
class CustomCNN(nn.Module):
    """Сверточная сеть с возможностью инъекции кастомных слоев"""
    def __init__(self, use_se=False, use_custom_act=False, use_mixed_pool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.act1 = LearnableSwish() if use_custom_act else nn.ReLU()
        self.pool1 = MixedPooling2d(2, 2) if use_mixed_pool else nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.act2 = LearnableSwish() if use_custom_act else nn.ReLU()
        
        self.se_block = SEBlock(64) if use_se else nn.Identity()
        
        self.pool2 = MixedPooling2d(2, 2) if use_mixed_pool else nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.conv2(x)
        x = self.se_block(x)
        x = self.pool2(self.act2(x))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResNetVariant(nn.Module):
    """Сеть для тестирования вариантов Residual блоков"""
    def __init__(self, block_type='bottleneck'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        if block_type == 'bottleneck':
            self.layer = BottleneckBlock(32, 16)
            out_features = 64
        else:
            self.layer = WideResidualBlock(32, 64, widen_factor=2)
            out_features = 64
            
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(out_features * 4 * 4, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def run_custom_layers_experiments():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_cifar_loaders(batch_size=128)
    
    models = {
        'Baseline (Standard ReLU & MaxPool)': CustomCNN(),
        'CNN + Learnable Swish': CustomCNN(use_custom_act=True),
        'CNN + Mixed Pooling': CustomCNN(use_mixed_pool=True),
        'CNN + SE-Block (Attention)': CustomCNN(use_se=True)
    }
    
    histories = {}
    for name, model in models.items():
        print(f"\n--- Обучение: {name} ---")
        model = model.to(device)
        print(f"Параметров: {count_parameters(model):,}")

        histories[name] = train_model(model, train_loader, test_loader, epochs=6, lr=0.002, device=device)
        
        if 'Swish' in name:
            print(f"Выученный параметр Beta у Swish: {model.act1.beta.item():.4f}")
        if 'Mixed Pooling' in name:
            # Пропускаем alpha через sigmoid, чтобы понять баланс между Max(1) и Avg(0)
            weight = torch.sigmoid(model.pool1.alpha).item()
            print(f"Баланс пулинга (1=Max, 0=Avg): {weight:.4f}")
            
    plot_comparison(histories, "Custom Layers Comparison", "custom_layers_comparison.png")

def run_residual_variants_experiments():
    print("\n=== Задание 3.2: Эксперименты с вариантами Residual блоков ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_cifar_loaders(batch_size=128)
    
    models = {
        'Bottleneck ResNet': ResNetVariant('bottleneck'),
        'Wide ResNet': ResNetVariant('wide')
    }
    
    histories = {}
    for name, model in models.items():
        print(f"\n--- Обучение: {name} ---")
        model = model.to(device)
        print(f"Параметров: {count_parameters(model):,}")
        histories[name] = train_model(model, train_loader, test_loader, epochs=6, lr=0.002, device=device)
        
    plot_comparison(histories, "Residual Variants Comparison", "residual_variants_comparison.png")

if __name__ == '__main__':
    run_custom_layers_experiments()
    run_residual_variants_experiments()