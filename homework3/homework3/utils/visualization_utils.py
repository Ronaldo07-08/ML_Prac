import matplotlib.pyplot as plt
import torch.nn as nn
import os
import numpy as np

import numpy as np
def plot_accuracy_comparison(histories, filename, title="Сравнение точности на Test"):
    """Сравнивает точность разных архитектур и сохраняет график."""
    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        if 'Regularization' not in name:
            plt.plot(history['test_accs'], marker='o', label=name)
            
    plt.title(title)
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (Accuracy)')
    plt.legend()
    plt.grid(True)
    
    path = os.path.join('homework', 'plots')
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    print(f"[*] График сохранен в: {os.path.join(path, filename)}")
    plt.show()

def plot_overfitting_analysis(history_overfit, history_reg, filename):
    """Визуализирует эффект регуляризации на переобучение."""
    plt.figure(figsize=(12, 6))
    plt.plot(history_overfit['train_accs'], 'r--', label='Train (Без регуляции)')
    plt.plot(history_overfit['test_accs'], 'r-', label='Test (Без регуляции - Переобучение)')
    
    plt.plot(history_reg['train_accs'], 'g--', label='Train (С регуляцией)')
    plt.plot(history_reg['test_accs'], 'g-', label='Test (С регуляцией - Стабильно)')
    
    plt.title('Влияние регуляризации на глубокую сеть')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (Accuracy)')
    plt.legend()
    plt.grid(True)
    
    path = os.path.join('homework', 'plots')
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    print(f"[*] График сохранен в: {os.path.join(path, filename)}")
    plt.show()

def plot_architecture_heatmap(results_matrix, row_labels, col_labels, filename, title="Heatmap"):
    """Визуализирует результаты Grid Search в виде тепловой карты."""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(results_matrix, cmap='viridis', aspect='auto')
    
    plt.xticks(np.arange(len(col_labels)), labels=col_labels)
    plt.yticks(np.arange(len(row_labels)), labels=row_labels)
    plt.colorbar(im, label='Точность (Accuracy)')

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            color = "black" if results_matrix[i, j] > np.max(results_matrix) - 0.02 else "white"
            plt.text(j, i, f"{results_matrix[i, j]:.4f}",
                     ha="center", va="center", color=color)

    plt.title(title)
    plt.xlabel("Базовый размер сети (N)")
    plt.ylabel("Схема изменения ширины")
    plt.tight_layout()

    path = os.path.join('homework', 'plots')
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"[*] Heatmap сохранен в: {filepath}")
    plt.show()

def plot_weight_distribution(models_dict, filename):
    """Строит гистограмму распределения весов первого линейного слоя."""
    plt.figure(figsize=(10, 6))
    
    for name, model in models_dict.items():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                
                weights = m.weight.detach().cpu().numpy().flatten()
                plt.hist(weights, bins=100, alpha=0.5, label=name, density=True)
                break 

    plt.title('Распределение весов первого скрытого слоя')
    plt.xlabel('Значение веса')
    plt.ylabel('Плотность')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = os.path.join('homework', 'plots')
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"[*] График весов сохранен в: {filepath}")
    plt.show()