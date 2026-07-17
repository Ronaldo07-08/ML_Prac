import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def analyze_dataset(root_dir):
    print(f"\n=== Задание 3: Анализ датасета ({root_dir}) ===")
    if not os.path.exists(root_dir):
        print(f"[!] Папка {root_dir} не найдена!")
        return

    class_counts = {}
    class_areas = {} 
    widths = []
    heights = []

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path): continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[class_name] = len(images)
        class_areas[class_name] = []

        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    class_areas[class_name].append(w * h)
            except Exception as e:
                print(f"Ошибка чтения {img_path}: {e}")

    # Вывод статистики в консоль
    print("\n--- Количество изображений по классам ---")
    for cls, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        print(f" - {cls}: {count}")

    print(f"\nВсего валидных изображений: {len(widths)}")
    if widths:
        print(f"Мин. размер: {min(widths)}x{min(heights)} px")
        print(f"Макс. размер: {max(widths)}x{max(heights)} px")
        print(f"Средний размер: {int(np.mean(widths))}x{int(np.mean(heights))} px")

    # Визуализация: Гистограммы площадей по каждому классу
    os.makedirs('homework5/results', exist_ok=True)

    for class_name, areas in class_areas.items():
        if not areas: continue
            
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(areas, bins=10, color='red', edgecolor='black')
        
        plt.title(f'Распределение площади по классу "{class_name}"')
        plt.xlabel('Площадь (пиксели²)')
        plt.ylabel('Количество изображений')
        
        plt.grid(True, alpha=0.3)

        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'homework5/results/task3_area_hist_{class_name}.png')
        plt.close()
        
    print(f"\n[*] Сгенерировано {len(class_areas)} гистограмм площадей по классам.")

    # Визуализация: Scatter plot размеров 
    os.makedirs('homework5/results', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='mediumpurple')
    plt.title('Распределение изображений по классам (Train)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Количество')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('homework5/results/task3_class_distribution.png')
    plt.close()

    if widths:
        plt.figure(figsize=(8, 6))
        plt.scatter(widths, heights, alpha=0.5, color='coral')
        plt.title('Распределение исходных размеров изображений')
        plt.xlabel('Ширина (px)')
        plt.ylabel('Высота (px)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('homework5/results/task3_size_scatter.png')
        plt.close()

    print("\n[*] Графики анализа сохранены в папку homework5/results/")

if __name__ == '__main__':
    analyze_dataset('data/train')