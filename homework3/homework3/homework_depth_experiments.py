import torch
import time
import os

from fully_connected_basics.datasets import get_mnist_loaders
from fully_connected_basics.models import FullyConnectedModel
from fully_connected_basics.trainer import train_model

from utils.model_utils import create_layer_config
from utils.experiment_utils import save_experiment_results
from utils.visualization_utils import plot_accuracy_comparison, plot_overfitting_analysis

# Задание 1.1: Сравнение моделей разной глубины
def run_depth_experiments():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используем устройство: {device}")
    
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Конфигурации для экспериментов: {Название: число скрытых слоев}
    # 1 слой (0 скрытых), 2 слоя (1 скрытый), 3 слоя (2), 5 слоев (4), 7 слоев (6)
    depth_configs = {
        '1 Layer (Linear)': 0,
        '2 Layers': 1,
        '3 Layers': 2,
        '5 Layers': 4,
        '7 Layers': 6
    }
    
    histories = {}
    training_times = {}
    
    for name, hidden_layers in depth_configs.items():
        print(f"\n--- Обучение модели: {name} ---")
        config = create_layer_config(hidden_layers, hidden_size=128)
        
        model = FullyConnectedModel(
            input_size=784, 
            num_classes=10, 
            layers=config
        ).to(device)
        
        start_time = time.time()
        # Обучаем 10 эпох 
        history = train_model(model, train_loader, test_loader, epochs=10, lr=0.002, device=device)
        end_time = time.time()
        
        histories[name] = history
        training_times[name] = end_time - start_time
        print(f"Время обучения: {training_times[name]:.2f} сек.")

    save_experiment_results(histories, 'depth_histories.json', subfolder='depth_experiments')
    save_experiment_results(training_times, 'depth_times.json', subfolder='depth_experiments')

    return histories, training_times

# Задание 1.2: Анализ переобучения и Регуляризация
def analyze_overfitting(histories):
    print("\n=== Задание 1.2: Анализ переобучения и Регуляризация ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    print("\nОбучаем глубокую сеть (7 слоев) С регуляризацией (BatchNorm + Dropout)...")
    config_reg = create_layer_config(num_hidden_layers=6, hidden_size=128, use_reg=True)
    
    model_reg = FullyConnectedModel(
        input_size=784, 
        num_classes=10, 
        layers=config_reg
    ).to(device)
      
    history_reg = train_model(model_reg, train_loader, test_loader, epochs=10, lr=0.002, device=device)
    histories['7 Layers (with Regularization)'] = history_reg
    
    save_experiment_results(histories, 'depth_histories_with_reg.json', subfolder='depth_experiments')

    return histories


if __name__ == '__main__':
    #  Запуск экспериментов с глубиной
    histories, times = run_depth_experiments()
    
    #  Анализ переобучения
    histories_full = analyze_overfitting(histories)
    

    plot_accuracy_comparison(histories_full, 'depth_test_accuracy.png')
    
    plot_overfitting_analysis(
        history_overfit=histories_full['7 Layers'], 
        history_reg=histories_full['7 Layers (with Regularization)'], 
        filename='overfitting_analysis.png'
    )
    
    print("\n--- Время обучения разных моделей ---")