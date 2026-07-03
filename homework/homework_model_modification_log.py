import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

class MulticlassDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MulticlassLogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

def run_multiclass_experiment():
    
    # Генерируем датасет на 3 класса
    X, y = make_classification(
        n_samples=300, 
        n_features=4, 
        n_informative=3,
        n_redundant=0,
        n_classes=3, 
        random_state=42
    )
    
    # Разбиваем на train, val
    X_train, y_train = X[:240], y[:240]
    X_val, y_val = X[240:], y[240:]

    dataset = MulticlassDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4 признака на входе, 3 класса на выходе
    model = MulticlassLogisticRegression(in_features=4, num_classes=3)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    epochs = 50
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Эпоха {epoch}: Train Loss = {total_loss/len(dataloader):.4f}")

    # Валидация и метрики
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32))
        
        val_probs = torch.softmax(val_logits, dim=1)
        
        val_preds = torch.argmax(val_probs, dim=1)

    # Переводим тензоры в numpy для работы sklearn
    y_val_np = y_val
    val_preds_np = val_preds.numpy()
    val_probs_np = val_probs.numpy()

    # Precision, Recall, F1. 
    #  считаем метрику для каждого класса отдельно, а потом берем среднее
    precision, recall, f1, _ = precision_recall_fscore_support(y_val_np, val_preds_np, average='macro', zero_division=0)

    # Для ROC-AUC многоклассовой классификации нужен параметр multi_class='ovr'
    roc_auc = roc_auc_score(y_val_np, val_probs_np, multi_class='ovr')

    print("\n--- Итоговые метрики на валидации ---")
    print(f"Precision (Точность): {precision:.4f}")
    print(f"Recall (Полнота):     {recall:.4f}")
    print(f"F1-score (Баланс):    {f1:.4f}")
    print(f"ROC-AUC:              {roc_auc:.4f}")

    cm = confusion_matrix(y_val_np, val_preds_np)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # график
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Матрица ошибок (Confusion Matrix)")
    plt.show()
    os.makedirs('homework/models', exist_ok=True)
    torch.save(model.state_dict(), 'homework/models/logreg_torch_modified.pth')

if __name__ == '__main__':
    run_multiclass_experiment()
    print("\nМодель успешно сохранена в 'homework/models/logreg_torch_modified.pth'")