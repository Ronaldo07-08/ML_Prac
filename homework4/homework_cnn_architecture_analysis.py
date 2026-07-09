import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import matplotlib.pyplot as plt

if os.path.exists('homework/convolutional_basics'):
    sys.path.insert(0, 'homework/convolutional_basics')
elif os.path.exists('convolutional_basics'):
    sys.path.insert(0, 'convolutional_basics')
sys.path.insert(0, 'homework')

from datasets import get_cifar_loaders
from convolutional_basics.trainer import train_model
from convolutional_basics.utils import count_parameters

from utils.visualization_utils import plot_comparison, visualize_feature_maps
from models.cnn_models import KernelCNN, InceptionLiteCNN, VariableDepthCNN

os.makedirs('homework/plots', exist_ok=True)
os.makedirs('homework/results/architecture_analysis', exist_ok=True)

def run_kernel_experiments():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_cifar_loaders(batch_size=128)
    
    # Чтобы количество параметров было сопоставимым, 
    # для больших ядер (5x5, 7x7) мы уменьшаем количество каналов.
    models = {
        'Kernel 3x3': KernelCNN(kernel_size=3, channels=32),
        'Kernel 5x5': KernelCNN(kernel_size=5, channels=16),
        'Kernel 7x7': KernelCNN(kernel_size=7, channels=10),
        'Inception (1x1 + 3x3)': InceptionLiteCNN(channels=16)
    }
    
    histories = {}
    for name, model in models.items():
        print(f"\n--- Обучение: {name} ---")
        model = model.to(device)
        print(f"Параметров: {count_parameters(model):,}")
        
        history = train_model(model, train_loader, test_loader, epochs=6, lr=0.002, device=device)
        histories[name] = history
        
        # Визуализируем, как каждый тип ядра видит картинку
        visualize_feature_maps(model, test_loader, device, name, f"feature_maps_{name.replace(' ', '_')}.png")
        
    plot_comparison(histories, "Kernel Size Comparison", "kernel_size_comparison.png")


def run_depth_experiments():
    print("\n=== Задание 2.2: Влияние глубины CNN ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_cifar_loaders(batch_size=128)
    
    models = {
        'Shallow (2 Conv)': VariableDepthCNN(num_layers=2),
        'Medium (4 Conv)': VariableDepthCNN(num_layers=4),
        'Deep (6 Conv)': VariableDepthCNN(num_layers=6)
    }
    
    histories = {}
    for name, model in models.items():
        print(f"\n--- Обучение: {name} ---")
        model = model.to(device)
        print(f"Параметров: {count_parameters(model):,}")
        
        history = train_model(model, train_loader, test_loader, epochs=8, lr=0.002, device=device)
        histories[name] = history
        
    plot_comparison(histories, "CNN Depth Comparison", "cnn_depth_comparison.png")

if __name__ == '__main__':
    run_kernel_experiments()
    run_depth_experiments()