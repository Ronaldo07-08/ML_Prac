import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.metrics import accuracy_score, mean_squared_error
import os


class CustomCSVDataset(Dataset):
    """
    Универсальный класс для загрузки и предобработки данных из CSV файла.
    """
    def __init__(self, csv_path: str, target_col: str, task_type: str = 'regression'):
        """
        Args:
            csv_path: Путь к файлу CSV
            target_col: Имя колонки, которую мы предсказываем
            task_type: 'regression' или 'classification'
        """
        print(f"Загрузка данных из {csv_path}...")
        df = pd.read_csv(csv_path)
        
        df = df.dropna()
        
        # Разделяем признаки
        self.y_data = df[target_col].values
        X_df = df.drop(columns=[target_col])
        
        cat_cols = X_df.select_dtypes(include=['object', 'category']).columns
        num_cols = X_df.select_dtypes(include=['number']).columns
        
        # Кодируем текстовые категории
        if len(cat_cols) > 0:
            X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
            X_df = X_df.astype(float)
            
        # Нормализуем числовые признаки
        if len(num_cols) > 0:
            scaler = StandardScaler()
            X_df[num_cols] = scaler.fit_transform(X_df[num_cols])
            
        self.X = torch.tensor(X_df.values, dtype=torch.float32)
        
        if task_type == 'regression':
            self.y = torch.tensor(self.y_data, dtype=torch.float32).unsqueeze(1)
        elif task_type == 'classification':
            self.y = torch.tensor(self.y_data, dtype=torch.float32).unsqueeze(1)
            
        self.input_dim = self.X.shape[1]
        print(f"Загружено строк: {len(self.X)}. Признаков после обработки: {self.input_dim}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_mock_csv_files():
    """Скачивает CSV файлы для тестов (Калифорнийские дома и Титаник)."""
    if not os.path.exists('data'):
        os.makedirs('data')
        
    if not os.path.exists('data/regression_data.csv'):
        cali = fetch_california_housing(as_frame=True)
        cali.frame.to_csv('data/regression_data.csv', index=False)
        
    if not os.path.exists('data/classification_data.csv'):
        titanic = fetch_openml(name='titanic', version=1, as_frame=True)
        df = titanic.frame
        cols_to_keep = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']
        df = df[cols_to_keep].dropna()
        df.to_csv('data/classification_data.csv', index=False)

def train_and_evaluate(model, dataloader, task_type='regression', epochs=20, lr=0.01):
    """Универсальный цикл обучения для регрессии и бинарной классификации."""
    if task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    model.eval()
    with torch.no_grad():
        all_X = dataloader.dataset.X
        all_y = dataloader.dataset.y
        preds = model(all_X)
        
        if task_type == 'regression':
            final_loss = criterion(preds, all_y).item()
            print(f"Обучение завершено. Итоговый MSE Loss: {final_loss:.4f}\n")
        else:
            probs = torch.sigmoid(preds)
            pred_classes = (probs > 0.5).float()
            acc = accuracy_score(all_y.numpy(), pred_classes.numpy())
            print(f"Обучение завершено. Итоговая Accuracy (Точность): {acc:.4f} ({acc*100:.1f}%)\n")


#---------- 2.2 Эксперименты с датасетами --------
def run_dataset_experiments():
    prepare_mock_csv_files()
    
    print("\n--- Эксперимент 1: Регрессия (Цены на жилье) ---")
    reg_dataset = CustomCSVDataset(
        csv_path='data/regression_data.csv', 
        target_col='MedHouseVal', 
        task_type='regression'
    )
    reg_dataloader = DataLoader(reg_dataset, batch_size=64, shuffle=True)
    
    # Модель - простая линейная регрессия
    lin_reg_model = nn.Linear(reg_dataset.input_dim, 1)
    train_and_evaluate(lin_reg_model, reg_dataloader, task_type='regression', epochs=30, lr=0.05)
    
    print("--- Эксперимент 2: Бинарная классификация (Выживание на Титанике) ---")
    clf_dataset = CustomCSVDataset(
        csv_path='data/classification_data.csv', 
        target_col='survived',
        task_type='classification'
    )
    clf_dataloader = DataLoader(clf_dataset, batch_size=32, shuffle=True)
    
    # Модель - логистическая регрессия
    log_reg_model = nn.Linear(clf_dataset.input_dim, 1)
    train_and_evaluate(log_reg_model, clf_dataloader, task_type='classification', epochs=30, lr=0.01)

if __name__ == '__main__':
    run_dataset_experiments()