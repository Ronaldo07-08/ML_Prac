import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_regression_data, mse, log_epoch, RegressionDataset
import os

class EarlyStopping:
    """
    Класс для реализации ранней остановки (Early Stopping).
    Останавливает обучение, если метрика на валидации не улучшается.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    X, y = make_regression_data(n=250)
    
    # Разбиваем данные на Train и Val 
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:], y[200:]
    
    # Создаём датасет и даталоадер только для обучающей выборки
    dataset = RegressionDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f'Размер обучающего датасета: {len(dataset)}')
    print(f'Размер валидационного датасета: {len(X_val)}')
    print(f'Количество батчей: {len(dataloader)}\n')
    

    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    l1_lambda = 0.01
    l2_lambda = 0.01
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    

    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            
            base_loss = criterion(y_pred, batch_y)
            
            # Вычисление штрафов L1 и L2
            l1_penalty = 0.0
            l2_penalty = 0.0
            for param in model.parameters():
                l1_penalty += torch.norm(param, p=1)
                l2_penalty += torch.norm(param, p=2) ** 2
            
            # Добавляем штрафы к итоговой функции потерь
            loss = base_loss + (l1_lambda * l1_penalty) + (l2_lambda * l2_penalty)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(dataloader)
        

        model.eval() # Переводим модель в режим оценки 
        with torch.no_grad(): # Отключаем слежку за градиентами 
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_train_loss, val_loss=val_loss)
            
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f'\n[!] Сработал Early Stopping на эпохе {epoch}. Ошибка на валидации перестала падать.')
            break

    os.makedirs('homework/models', exist_ok=True)   
    # Сохраняем модифицированную модель
    torch.save(model.state_dict(), 'homework/models/linreg_torch_modified.pth')
    print("\nМодель успешно сохранена в 'homework/models/linreg_torch_modified.pth'")