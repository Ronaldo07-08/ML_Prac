import torch
import time
import os
import sys
import numpy as np


from fully_connected_basics.datasets import get_mnist_loaders
from fully_connected_basics.models import FullyConnectedModel
from fully_connected_basics.trainer import train_model
from fully_connected_basics.utils import count_parameters

from utils.model_utils import create_custom_width_config
from utils.experiment_utils import save_experiment_results
from utils.visualization_utils import plot_accuracy_comparison, plot_architecture_heatmap

def run_width_experiments():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Конфигурации
    width_configs = {
        'Narrow [64, 32, 16]': [64, 32, 16],
        'Medium [256, 128, 64]': [256, 128, 64],
        'Wide [1024, 512, 256]': [1024, 512, 256],
        'Very Wide [2048, 1024, 512]': [2048, 1024, 512]
    }
    
    histories = {}
    training_times = {}
    param_counts = {}
    
    for name, layer_sizes in width_configs.items():
        print(f"\n--- Обучение модели: {name} ---")
        config = create_custom_width_config(layer_sizes)
        
        model = FullyConnectedModel(
            input_size=784, 
            num_classes=10, 
            layers=config
        ).to(device)
        
        p_count = count_parameters(model)
        param_counts[name] = p_count
        print(f"Количество параметров: {p_count:,}")
        
        start_time = time.time()
        # Обучаем 5 эпох
        history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)
        end_time = time.time()
        
        histories[name] = history
        training_times[name] = end_time - start_time
        print(f"Время обучения: {training_times[name]:.2f} сек.")

    save_experiment_results(histories, 'width_histories.json', subfolder='width_experiments')
    save_experiment_results(training_times, 'width_times.json', subfolder='width_experiments')
    save_experiment_results(param_counts, 'width_params.json', subfolder='width_experiments')

    return histories, training_times, param_counts

def run_architecture_optimization():
    print("\n=== Задание 2.2: Оптимизация архитектуры (Grid Search) ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_mnist_loaders(batch_size=256)
    
    # Схемы изменения ширины и базовые размеры
    schemes = ['Сужающаяся', 'Постоянная', 'Расширяющаяся']
    base_sizes = [64, 256, 512]
    
    # Матрица для хранения лучшей точности
    results_matrix = np.zeros((len(schemes), len(base_sizes)))
    
    for i, scheme in enumerate(schemes):
        for j, N in enumerate(base_sizes):
            if scheme == 'Сужающаяся':
                layer_sizes = [N, max(N//2, 16), max(N//4, 16)]
            elif scheme == 'Постоянная':
                layer_sizes = [N, N, N]
            else:
                layer_sizes = [max(N//4, 16), max(N//2, 16), N]
                
            print(f"\nТест: Схема='{scheme}', База N={N}, Слои={layer_sizes}")
            config = create_custom_width_config(layer_sizes)
            model = FullyConnectedModel(input_size=784, num_classes=10, layers=config).to(device)
            
            history = train_model(model, train_loader, test_loader, epochs=3, lr=0.002, device=device)
            
            # Берем максимальную точность на тесте
            best_acc = max(history['test_accs'])
            results_matrix[i, j] = best_acc
            
    return results_matrix, schemes, base_sizes

if __name__ == '__main__':
    # Запуск экспериментов с шириной
    histories, times, params = run_width_experiments()
    
    # Сравнительный график
    plot_accuracy_comparison(histories, 'width_test_accuracy.png', title='Сравнение точности от ширины сети')
    
    # Аналитика времени и веса
    print("\n--- Анализ количества параметров и времени ---")
    for name in histories.keys():
        print(f"{name} -> Параметров: {params[name]:,}, Время: {times[name]:.2f} сек.")
        
    # Поиск оптимальной архитектуры
    matrix, schemes, base_sizes = run_architecture_optimization()
    
    # Heatmap
    plot_architecture_heatmap(matrix, schemes, base_sizes, 'architecture_heatmap.png', title='Grid Search: Оптимизация архитектуры')