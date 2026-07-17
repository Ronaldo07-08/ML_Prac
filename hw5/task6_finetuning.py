import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import matplotlib.pyplot as plt

if os.path.exists('augmentations_basics'):
    sys.path.insert(0, 'augmentations_basics')
sys.path.insert(0, 'homework5')

from datasets import CustomImageDataset

def get_dataloaders(batch_size=32, val_split=0.2):
    """
    Загружает датасет из папки train и разбивает его на train и val.
    Настраивает разные трансформации для обеих частей.
    """
    
    train_dir = 'data/train'
    
    if not os.path.exists(train_dir):
        print(f"[!] Ошибка: Папка {train_dir} не найдена.")
        return None, None, None

    # Для предобученных моделей (как ResNet) важно использовать Normalize
    # со средними и стандартными отклонениями ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 1. Трансформации для тренировки (с аугментациями)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])

    # 2. Трансформации для валидации (только ресайз и нормализация, без искажений)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Сначала загружаем весь датасет без трансформаций
    full_dataset = CustomImageDataset(train_dir, transform=None, target_size=(224, 224))
    class_names = full_dataset.get_class_names()
    
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # Разбиваем датасет
    train_subset, val_subset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # Фиксируем seed для воспроизводимости
    )
    
    # У CustomImageDataset трансформы задаются глобально для всего датасета.
    # Так как мы разбили один датасет на два Subset, мы не можем просто так назначить им разные трансформы.
    # Поэтому мы создаем класс-обертку, который применяет трансформации на лету.
    class DatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            # В CustomImageDataset без transform __getitem__ возвращает (PIL Image, label)
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
            
        def __len__(self):
            return len(self.subset)

    train_dataset = DatasetWrapper(train_subset, transform=train_transform)
    val_dataset = DatasetWrapper(val_subset, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[*] Датасет разделен: {train_size} изображений для обучения, {val_size} для валидации.")
    
    return train_loader, val_loader, class_names

def build_model(num_classes, device):
    """Загружает ResNet18 и адаптирует ее под наше количество классов."""
    print("[*] Загрузка предобученной модели ResNet18...")
    # Загружаем модель с весами ImageNet
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Замораживаем веса всех слоев, чтобы они не разрушились в начале обучения
    for param in model.parameters():
        param.requires_grad = False
        
    # Заменяем последний полносвязный слой.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model.to(device)

def train_model(model, train_loader, val_loader, device, epochs=5):
    """Цикл обучения с отслеживанием loss и accuracy."""
    
    # Обучаем только размороженные параметры (последний слой)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n[*] Старт обучения на {epochs} эпох...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- ФАЗА ОБУЧЕНИЯ (TRAIN) ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # --- ФАЗА ВАЛИДАЦИИ (VAL) ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        # Сохраняем в историю
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        elapsed = time.time() - start_time
        print(f"Эпоха {epoch+1}/{epochs} [{elapsed:.1f}s] "
              f"| Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} "
              f"| Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
              
    return history

def plot_and_save_history(history, save_path):
    """Визуализирует кривые потерь и точности."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # График Loss
    ax1.plot(epochs, history['train_loss'], color='royalblue', marker='o', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color='crimson', marker='s', linewidth=2, label='Val Loss')
    ax1.set_title('График потерь (Loss)')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График Accuracy
    ax2.plot(epochs, history['train_acc'], color='royalblue', marker='o', linewidth=2, label='Train Acc')
    ax2.plot(epochs, history['val_acc'], color='crimson', marker='s', linewidth=2, label='Val Acc')
    ax2.set_title('График точности (Accuracy)')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n[*] Графики обучения сохранены в {save_path}")

def run_finetuning():
    print("=== Задание 6: Дообучение предобученных моделей (ResNet18) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Используемое устройство: {device}")
    
    train_loader, val_loader, class_names = get_dataloaders(batch_size=32, val_split=0.2)
    if train_loader is None: return
    
    num_classes = len(class_names)
    print(f"[*] Найдено классов: {num_classes} ({', '.join(class_names)})")
    
    model = build_model(num_classes, device)
    
    # Обучаем 5 эпох
    history = train_model(model, train_loader, val_loader, device, epochs=5)
    
    # Сохраняем графики
    plot_and_save_history(history, 'homework5/results/task6_finetuning.png')

if __name__ == '__main__':
    run_finetuning()