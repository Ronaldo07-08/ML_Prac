import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import logging
import numpy as np
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Базовая модель и цикл обучения для экспериментов
class SimpleLinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

def train_experiment(X: torch.Tensor, y: torch.Tensor, 
                    batch_size: int = 32, lr: float = 0.01, 
                    optimizer_name: str = 'SGD', epochs: int = 50) -> list:
    """
    Универсальная функция для обучения модели с заданными гиперпараметрами.
    
    Returns:
        loss_history (list): История изменения функции потерь по эпохам.
    """
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleLinearRegression(in_features=X.shape[1])
    criterion = nn.MSELoss()
    
    # Выбор оптимизатора
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")

    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
    return loss_history

# 3.1 Исследование гиперпараметров
def run_hyperparameter_experiments():
    logger.info("Начало Эксперимента 3.1: Исследование гиперпараметров")
    
    # Генерим линейную зависимость
    torch.manual_seed(42)
    X = torch.randn(500, 3)
    true_w = torch.tensor([[2.0], [-1.5], [0.8]])
    y = X @ true_w + 0.5 + torch.randn(500, 1) * 0.1

    configs = [
        {'opt': 'SGD', 'lr': 0.01, 'bs': 32},
        {'opt': 'SGD', 'lr': 0.1, 'bs': 32}, 
        {'opt': 'Adam', 'lr': 0.1, 'bs': 32},
        {'opt': 'RMSprop', 'lr': 0.05, 'bs': 64}
    ]
    
    plt.figure(figsize=(10, 6))
    
    for conf in configs:
        logger.info(f"Обучение: Оптимизатор={conf['opt']}, LR={conf['lr']}, Batch={conf['bs']}")
        history = train_experiment(X, y, batch_size=conf['bs'], lr=conf['lr'], 
                                   optimizer_name=conf['opt'], epochs=40)
        plt.plot(history, label=f"{conf['opt']} (LR={conf['lr']}, BS={conf['bs']})")

    plt.title("Сравнение скорости сходимости разных гиперпараметров")
    plt.xlabel("Эпохи")
    plt.ylabel("MSE Loss")
    plt.yscale('log') # Логарифмическая шкала
    plt.legend()
    plt.grid(True)
    os.makedirs('homework/plots', exist_ok=True) # Создаем папку
    plt.savefig('homework/plots/hyperparameters_comparison.png', bbox_inches='tight') # Сохраняем
    plt.show() # И только потом показываем

# 3.2 Feature Engineering
def run_feature_engineering_experiments():
    logger.info("Начало Эксперимента 3.2: Feature Engineering")
    
    # Генерим нелинейные данные
    torch.manual_seed(42)
    X_base = torch.randn(500, 2)
    # y = x1^2 + 2*x1*x2
    y = (X_base[:, 0]**2 + 2 * X_base[:, 0] * X_base[:, 1]).unsqueeze(1) + torch.randn(500, 1) * 0.1
    
    # Обучаем базовую модель на исходных данных
    logger.info("Обучение базовой модели ...")
    base_history = train_experiment(X_base, y, batch_size=32, lr=0.05, optimizer_name='Adam', epochs=100)
    final_base_loss = base_history[-1]
    
    # Добавляем новые признаки x1^2, x2^2
    feat_sq = X_base ** 2
    
    # Взаимодействия: x1 * x2
    feat_interact = (X_base[:, 0] * X_base[:, 1]).unsqueeze(1)
    
    # Статистические: среднее и дисперсия по строке
    feat_mean = X_base.mean(dim=1, keepdim=True)
    feat_var = X_base.var(dim=1, keepdim=True)
    
    X_engineered = torch.cat([X_base, feat_sq, feat_interact, feat_mean, feat_var], dim=1)
    
    logger.info(f"Количество признаков увеличено с {X_base.shape[1]} до {X_engineered.shape[1]}")
    
    # Обучаем модель на расширенных данных
    logger.info("Обучение продвинутой модели (с новыми признаками)...")
    eng_history = train_experiment(X_engineered, y, batch_size=32, lr=0.05, optimizer_name='Adam', epochs=100)
    final_eng_loss = eng_history[-1]
    
    logger.info(f"Итоговый MSE базовой модели: {final_base_loss:.4f}")
    logger.info(f"Итоговый MSE с Feature Engineering: {final_eng_loss:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(base_history, label=f"Базовая модель (Loss: {final_base_loss:.2f})", color='red')
    plt.plot(eng_history, label=f"Feature Engineering (Loss: {final_eng_loss:.2f})", color='green')
    plt.title("Влияние Feature Engineering на обучение модели")
    plt.xlabel("Эпохи")
    plt.ylabel("MSE Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    os.makedirs('homework/plots', exist_ok=True) # Создаем папку
    plt.savefig('homework/plots/feature_engineering_effect.png', bbox_inches='tight') # Сохраняем
    plt.show()

if __name__ == '__main__':
    run_hyperparameter_experiments()
    print("\n" + "="*50 + "\n")
    run_feature_engineering_experiments()