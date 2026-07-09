import torch
import time
import os
import sys

from datasets import get_mnist_loaders, get_cifar_loaders
from convolutional_basics.trainer import train_model
from convolutional_basics.utils import count_parameters

from models.fc_models import DeepFCNet
from models.cnn_models import SimpleCNN, CNNWithResidual

from utils.visualization_utils import plot_comparison

os.makedirs('homework/plots', exist_ok=True)
os.makedirs('homework/results/mnist_comparison', exist_ok=True)
os.makedirs('homework/results/cifar_comparison', exist_ok=True)

def run_mnist_comparison():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    models = {
        'FC Network (3 Layers)': DeepFCNet(input_size=28*28, num_classes=10),
        'Simple CNN (2 Conv)': SimpleCNN(input_channels=1, num_classes=10),
        'Residual CNN': CNNWithResidual(input_channels=1, num_classes=10)
    }
    
    histories = {}
    for name, model in models.items():
        print(f"\n--- Обучение: {name} ---")
        model = model.to(device)
        print(f"Количество параметров: {count_parameters(model):,}")
        
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)
        train_time = time.time() - start_time
        
        print(f"Время обучения: {train_time:.2f} сек.")
        histories[name] = history
        
    plot_comparison(histories, "MNIST Models Comparison", "mnist_cnn_vs_fc.png")

def run_cifar_comparison():
    print("\n=== Задание 1.2: Сравнение на CIFAR-10 ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_cifar_loaders(batch_size=128)
    
    models = {
        'Deep FC Network': DeepFCNet(input_size=32*32*3, num_classes=10),
        'Residual CNN (No Reg)': CNNWithResidual(input_channels=3, num_classes=10, use_dropout=False),
        'Residual CNN (With Reg)': CNNWithResidual(input_channels=3, num_classes=10, use_dropout=True)
    }
    
    histories = {}
    for name, model in models.items():
        print(f"\n--- Обучение: {name} ---")
        model = model.to(device)
        print(f"Количество параметров: {count_parameters(model):,}")
        
        start_time = time.time()
        # Для CIFAR нужно чуть больше эпох, так как задача сложнее
        history = train_model(model, train_loader, test_loader, epochs=8, lr=0.001, device=device)
        train_time = time.time() - start_time
        
        print(f"Время обучения: {train_time:.2f} сек.")
        histories[name] = history
        
    plot_comparison(histories, "CIFAR-10 Models Comparison", "cifar_cnn_vs_fc.png")

if __name__ == '__main__':
    run_mnist_comparison()
    run_cifar_comparison()