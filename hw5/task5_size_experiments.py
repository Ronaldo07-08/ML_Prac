import os
import sys
import time
import tracemalloc
import matplotlib.pyplot as plt
from torchvision import transforms

if os.path.exists('augmentations_basics'):
    sys.path.insert(0, 'augmentations_basics')
sys.path.insert(0, 'homework5')

from datasets import CustomImageDataset

def run_size_experiments():
    print("=== Задание 5: Эксперимент с размерами изображений ===")
    
    root_dir = 'data/train'
    if not os.path.exists(root_dir):
        print(f"[!] Папка {root_dir} не найдена!")
        return
        
    sizes = [(64, 64), (128, 128), (224, 224), (512, 512)]
    size_labels = [f"{s[0]}x{s[1]}" for s in sizes]
    
    times = []
    memories = []
    
    # Общий пайплайн аугментаций для чистоты эксперимента
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor()
    ])
    
    num_images_to_test = 100
    
    for size in sizes:
        print(f"[*] Тестирование размера {size[0]}x{size[1]}...")
        dataset = CustomImageDataset(root_dir, transform=transform, target_size=size)
        
        # Ограничиваем количество изображений до 100
        actual_num = min(num_images_to_test, len(dataset))
        
        # Включаем замер памяти
        tracemalloc.start()
        start_time = time.time()
        
        # Сохраняем в список, чтобы замерить общее потребление памяти
        batch = []
        for i in range(actual_num):
            img, label = dataset[i]
            batch.append(img)
            
        end_time = time.time()
        # Получаем пиковое потребление памяти
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Очищаем память от батча для следующей итерации
        del batch
        
        elapsed_time = end_time - start_time
        peak_memory_mb = peak_memory / (1024 * 1024)
        
        times.append(elapsed_time)
        memories.append(peak_memory_mb)
        
        print(f"    Время: {elapsed_time:.2f} сек. | Память: {peak_memory_mb:.2f} MB")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # График времени
    ax1.plot(size_labels, times, marker='o', color='royalblue', linewidth=2)
    ax1.set_title(f'Зависимость времени от размера\n(Обработка {actual_num} изображений)')
    ax1.set_xlabel('Размер изображения')
    ax1.set_ylabel('Время (секунды)')
    ax1.grid(True, alpha=0.3)
    
    # График памяти
    ax2.plot(size_labels, memories, marker='s', color='crimson', linewidth=2)
    ax2.set_title(f'Зависимость потребления памяти от размера\n(Хранение {actual_num} изображений)')
    ax2.set_xlabel('Размер изображения')
    ax2.set_ylabel('Пиковая память (MB)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('homework5/results', exist_ok=True)
    save_path = 'homework5/results/task5_size_experiment.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n[*] Графики экспериментов сохранены в {save_path}")

if __name__ == '__main__':
    run_size_experiments()