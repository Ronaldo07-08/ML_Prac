import os
import sys
import random
from torchvision import transforms

if os.path.exists('augmentations_basics'):
    sys.path.insert(0, 'augmentations_basics')
sys.path.insert(0, 'homework5')

from datasets import CustomImageDataset
from utils.visualization import save_augmentation_grid

def get_five_random_images(dataset):
    """Выбирает 5 случайных изображений из разных классов."""
    # Группируем индексы по классам
    class_indices = {}
    for idx, label in enumerate(dataset.labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
        
    # Выбираем до 5 случайных классов
    selected_classes = random.sample(list(class_indices.keys()), min(5, len(class_indices)))
    
    selected_images = []
    for cls in selected_classes:
        idx = random.choice(class_indices[cls])
        img, _ = dataset[idx]
        selected_images.append(img)
        
    return selected_images

def run_standard_augmentations():
    
    root_dir = 'data/train'
    if not os.path.exists(root_dir):
        print(f"Ошибка: папка {root_dir} не найдена.")
        return
        
    # Загружаем датасет (без аугментаций на этапе загрузки)
    dataset = CustomImageDataset(root_dir, transform=None, target_size=(224, 224))
    original_imgs = get_five_random_images(dataset)
    
    print(f"Отобрано {len(original_imgs)} изображений для тестирования.")

    # Стандартные аугментации torchvision
    standard_augs = {
        "H-Flip": transforms.RandomHorizontalFlip(p=1.0),
        "RandomCrop": transforms.RandomCrop(180, padding=20),
        "ColorJitter": transforms.ColorJitter(brightness=0.5, contrast=0.5),
        "Rotation (30)": transforms.RandomRotation(degrees=30),
        "Grayscale": transforms.RandomGrayscale(p=1.0)
    }
    
    combined_standard = transforms.Compose(list(standard_augs.values()))
    standard_augs["Combined Standard"] = combined_standard

    standard_results = {}
    for name, aug in standard_augs.items():
        standard_results[name] = [aug(img) for img in original_imgs]
        
    save_augmentation_grid(original_imgs, standard_results, 'task1_standard_augs.png')

if __name__ == '__main__':
    run_standard_augmentations()