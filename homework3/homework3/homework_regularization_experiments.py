import torch
import os
import sys

from fully_connected_basics.datasets import get_mnist_loaders
from fully_connected_basics.models import FullyConnectedModel

from utils.model_utils import build_reg_config
from utils.experiment_utils import save_experiment_results, custom_train_loop
from utils.visualization_utils import plot_accuracy_comparison, plot_weight_distribution

def run_regularization_experiments():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Чтобы увидеть эффект переобучения, уменьшим батч 
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    experiments = {
        'No Regularization': {'reg': 'none', 'dr': 0.0, 'wd': 0.0},
        'Dropout (0.3)': {'reg': 'dropout', 'dr': 0.3, 'wd': 0.0},
        'Dropout (0.5)': {'reg': 'dropout', 'dr': 0.5, 'wd': 0.0},
        'BatchNorm': {'reg': 'batchnorm', 'dr': 0.0, 'wd': 0.0},
        'BatchNorm + Dropout': {'reg': 'both', 'dr': 0.3, 'wd': 0.0},
        'L2 (Weight Decay=1e-3)': {'reg': 'none', 'dr': 0.0, 'wd': 1e-3}
    }
    
    histories = {}
    trained_models = {}
    
    for name, params in experiments.items():
        print(f"\n--- Обучение: {name} ---")
        config = build_reg_config(sizes=[256, 128, 64], reg_type=params['reg'], dropout_rate=params['dr'])
        
        model = FullyConnectedModel(input_size=784, num_classes=10, layers=config).to(device)
        history = custom_train_loop(
            model, train_loader, test_loader, 
            epochs=8, lr=0.001, 
            weight_decay=params['wd'], device=device
        )
        histories[name] = history
        trained_models[name] = model

    save_experiment_results(histories, 'reg_histories.json', subfolder='regularization_experiments')
    
    plot_accuracy_comparison(histories, 'reg_test_accuracy.png', title='Влияние регуляризации на точность (Test)')
    
    models_to_plot = {
        'Без регуляризации': trained_models['No Regularization'],
        'L2 (Weight Decay)': trained_models['L2 (Weight Decay=1e-3)'],
        'BatchNorm': trained_models['BatchNorm']
    }
    plot_weight_distribution(models_to_plot, 'weight_distributions.png')

    return histories

def run_adaptive_regularization():
    print("\n=== Задание 3.2: Адаптивная регуляризация ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    config = build_reg_config(sizes=[256, 128, 64], reg_type='both', dropout_rate=0.1) # Начинаем с малого Dropout
    model = FullyConnectedModel(input_size=784, num_classes=10, layers=config).to(device)
    
    print("Запуск адаптивного обучения (Dropout растет, Momentum падает)...")
    history_adaptive = custom_train_loop(
        model, train_loader, test_loader, 
        epochs=10, lr=0.002, device=device, adaptive=True
    )
    
    histories = {'Adaptive Reg (Dynamic Dropout & BN)': history_adaptive}
    plot_accuracy_comparison(histories, 'adaptive_reg_accuracy.png', title='Точность адаптивной регуляризации')

if __name__ == '__main__':
    # Запуск классических экспериментов
    run_regularization_experiments()
    
    # Запуск адаптивных экспериментов
    run_adaptive_regularization()