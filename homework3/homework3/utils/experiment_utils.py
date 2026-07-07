import json
import os
import torch.nn as nn
import torch.optim as optim
from fully_connected_basics.trainer import run_epoch

def save_experiment_results(results_dict, filename, subfolder="depth_experiments"):
    """
    Сохраняет историю обучения в формате JSON в папку results.
    """
    path = os.path.join('homework', 'results', subfolder)
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, filename)
    with open(file_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"[*] Результаты сохранены в: {file_path}")

def custom_train_loop(model, train_loader, test_loader, epochs=10, lr=0.002, 
                      weight_decay=0.0, device='cpu', adaptive=False):
    """
    Кастомный цикл обучения с поддержкой Weight Decay (L2) и адаптивной регуляризации.
    """
    criterion = nn.CrossEntropyLoss()
    # Weight decay в PyTorch передается прямо в оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {'train_losses': [], 'train_accs': [], 'test_losses': [], 'test_accs': []}
    
    for epoch in range(epochs):
        # Если включен адаптивный режим, меняем гиперпараметры прямо на лету!
        if adaptive:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    # Постепенно увеличиваем вероятность выключения нейронов
                    m.p = min(0.6, m.p + 0.05) 
                elif isinstance(m, nn.BatchNorm1d):
                    # Постепенно уменьшаем momentum
                    m.momentum = max(0.01, m.momentum * 0.9)

        # Используем функции run_epoch из кода преподавателя
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        vl, va = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        
        history['train_losses'].append(tl)
        history['train_accs'].append(ta)
        history['test_losses'].append(vl)
        history['test_accs'].append(va)
        
        # Выводим прогресс адаптивных параметров
        if adaptive and epoch == epochs - 1:
            print("[Адаптивный режим] Финальные параметры:")
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    print(f"  Dropout p: {m.p:.2f}")
                    break
                    
    return history