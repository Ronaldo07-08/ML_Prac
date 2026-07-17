import os
import sys
import random

if os.path.exists('augmentations_basics'):
    sys.path.insert(0, 'augmentations_basics')
sys.path.insert(0, 'homework5')

from datasets import CustomImageDataset
from utils.custom_transforms import CustomRandomBlur, CustomColorJitter, CustomPixelate
from utils.visualization import save_augmentation_grid

def get_five_random_images(dataset):
    """Выбирает 5 случайных изображений из разных классов."""
    class_indices = {}
    for idx, label in enumerate(dataset.labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
        
    selected_classes = random.sample(list(class_indices.keys()), min(5, len(class_indices)))
    
    selected_images = []
    for cls in selected_classes:
        idx = random.choice(class_indices[cls])
        img, _ = dataset[idx]
        selected_images.append(img)
        
    return selected_images

def run_custom_augmentations():
    print("=== Задание 2: Кастомные Аугментации ===")
    
    root_dir = 'data/train'
    if not os.path.exists(root_dir):
        print(f"Ошибка: папка {root_dir} не найдена. Убедитесь, что датасет распакован.")
        return
        
    dataset = CustomImageDataset(root_dir, transform=None, target_size=(224, 224))
    original_imgs = get_five_random_images(dataset)
    
    print(f"Отобрано {len(original_imgs)} изображений для тестирования.")

    # Кастомные аугментации
    custom_augs = {
        "Custom Blur": CustomRandomBlur(p=1.0, radius=(2, 5)),
        "Custom Color": CustomColorJitter(p=1.0, brightness=2.0, contrast=2.0),
        "Custom Pixelate": CustomPixelate(p=1.0, pixel_size=8)
    }
    
    custom_results = {}
    for name, aug in custom_augs.items():
        custom_results[name] = [aug(img) for img in original_imgs]
        
    save_augmentation_grid(original_imgs, custom_results, 'task2_custom_augs.png')

if __name__ == '__main__':
    run_custom_augmentations()