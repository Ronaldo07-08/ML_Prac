import os
import sys
import torch
from torchvision import transforms
from PIL import Image

if os.path.exists('augmentations_basics'):
    sys.path.insert(0, 'augmentations_basics')
sys.path.insert(0, 'homework5')

from datasets import CustomImageDataset
from extra_augs import AddGaussianNoise, CutOut, Solarize, Posterize, AutoContrast
from utils.custom_transforms import CustomRandomBlur, CustomColorJitter
from utils.visualization import show_images
import matplotlib.pyplot as plt


class AugmentationPipeline:
    """Пайплайн для гибкой настройки и применения аугментаций."""
    
    def __init__(self):
        # Храним аугментации в виде словаря: {name: transform_function}
        # Используем словарь, чтобы легко удалять аугментации по имени.
        self.augmentations = {}

    def add_augmentation(self, name, aug):
        """Добавляет аугментацию в конец пайплайна."""
        self.augmentations[name] = aug
        print(f"[+] Добавлена аугментация: {name}")

    def remove_augmentation(self, name):
        """Удаляет аугментацию по имени."""
        if name in self.augmentations:
            del self.augmentations[name]
            print(f"[-] Удалена аугментация: {name}")
        else:
            print(f"[!] Аугментация '{name}' не найдена в пайплайне.")

    def get_augmentations(self):
        """Возвращает список текущих аугментаций."""
        return list(self.augmentations.keys())

    def apply(self, image):
        """
        Применяет все аугментации по очереди к изображению.
        Если аугментация возвращает тензор, а следующая ждет PIL Image 
        (или наоборот), это может вызвать ошибку. 
        Поэтому наш пайплайн сам следит за форматами.
        """
        img = image
        for name, aug in self.augmentations.items():
            try:
                
                # Если аугментация ждет Tensor, а у нас PIL:
                if isinstance(aug, (AddGaussianNoise, CutOut, Solarize, Posterize)) and not isinstance(img, torch.Tensor):
                    img = transforms.ToTensor()(img)
                    
                # Если аугментация ждет PIL, а у нас Tensor:
                elif isinstance(aug, (transforms.ColorJitter, CustomRandomBlur, CustomColorJitter)) and isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)
                
                img = aug(img)
            except Exception as e:
                print(f"[!] Ошибка при применении {name}: {e}")
                
        return img


def run_pipeline_experiments():
    print("=== Задание 4: Эксперименты с AugmentationPipeline ===\n")
    
    root_dir = 'data/train'
    if not os.path.exists(root_dir):
        print(f"[!] Папка {root_dir} не найдена!")
        return
        
    dataset = CustomImageDataset(root_dir, transform=None, target_size=(224, 224))
    original_img, _ = dataset[0]
    
    #  LIGHT
    print("--- Конфигурация LIGHT ---")
    light_pipeline = AugmentationPipeline()
    light_pipeline.add_augmentation("H-Flip", transforms.RandomHorizontalFlip(p=1.0))
    light_pipeline.add_augmentation("Light ColorJitter", transforms.ColorJitter(brightness=0.2, contrast=0.2))
    
    #  MEDIUM 
    print("\n--- Конфигурация MEDIUM ---")
    medium_pipeline = AugmentationPipeline()
    medium_pipeline.add_augmentation("H-Flip", transforms.RandomHorizontalFlip(p=1.0))
    medium_pipeline.add_augmentation("Rotation", transforms.RandomRotation(degrees=15))
    medium_pipeline.add_augmentation("Medium Blur", CustomRandomBlur(p=1.0, radius=(1, 2)))
    # Тестируем метод remove
    medium_pipeline.add_augmentation("Temp Grayscale", transforms.RandomGrayscale(p=1.0))
    medium_pipeline.remove_augmentation("Temp Grayscale")
    
    #  HEAVY
    print("\n--- Конфигурация HEAVY ---")
    heavy_pipeline = AugmentationPipeline()
    heavy_pipeline.add_augmentation("H-Flip", transforms.RandomHorizontalFlip(p=1.0))
    heavy_pipeline.add_augmentation("Rotation", transforms.RandomRotation(degrees=45))
    heavy_pipeline.add_augmentation("Heavy Blur", CustomRandomBlur(p=1.0, radius=(3, 5)))
    heavy_pipeline.add_augmentation("Noise", AddGaussianNoise(0., 0.2))
    heavy_pipeline.add_augmentation("CutOut", CutOut(p=1.0, size=(64, 64)))
    
    print("\n[*] Текущие слои HEAVY пайплайна:", heavy_pipeline.get_augmentations())

    # Применяем пайплайны
    light_result = light_pipeline.apply(original_img)
    medium_result = medium_pipeline.apply(original_img)
    heavy_result = heavy_pipeline.apply(original_img)
    
    # Если результат остался в PIL, переводим в Tensor для функции визуализации
    if not isinstance(light_result, torch.Tensor): light_result = transforms.ToTensor()(light_result)
    if not isinstance(medium_result, torch.Tensor): medium_result = transforms.ToTensor()(medium_result)
    if not isinstance(heavy_result, torch.Tensor): heavy_result = transforms.ToTensor()(heavy_result)
    original_tensor = transforms.ToTensor()(original_img)

    #  Визуализация 
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    images = [original_tensor, light_result, medium_result, heavy_result]
    titles = ["Original", "Light", "Medium", "Heavy"]
    
    for i, (img_tensor, title) in enumerate(zip(images, titles)):
        # Tensor -> Numpy для отрисовки
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = img_np.clip(0, 1)
        axes[i].imshow(img_np)
        axes[i].set_title(title, fontsize=14, fontweight='bold' if i==0 else 'normal')
        axes[i].axis('off')
        
    plt.tight_layout()
    os.makedirs('homework5/results', exist_ok=True)
    save_path = 'homework5/results/task4_pipeline_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n[*] Сравнение пайплайнов сохранено в {save_path}")

if __name__ == '__main__':
    run_pipeline_experiments()