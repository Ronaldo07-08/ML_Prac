import matplotlib.pyplot as plt
import os
from torchvision import transforms

def save_augmentation_grid(original_imgs, aug_dict, filename):
    """
    Создает сетку изображений: Оригинал + Все аугментации в ряд.
    original_imgs: список из PIL Images.
    aug_dict: словарь {"Название аугментации": [список измененных картинок]}.
    """
    n_rows = len(original_imgs)
    n_cols = 1 + len(aug_dict)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    # Защита от случая, когда n_rows = 1 (axes становится 1D массивом)
    if n_rows == 1:
        axes = [axes]

    for r in range(n_rows):
        # Отрисовка оригинала в первом столбце
        axes[r][0].imshow(original_imgs[r])
        if r == 0: 
            axes[r][0].set_title("Original", fontsize=14, fontweight='bold')
        axes[r][0].axis('off')

        # Отрисовка аугментаций в следующих столбцах
        for c, (aug_name, aug_imgs) in enumerate(aug_dict.items(), start=1):
            axes[r][c].imshow(aug_imgs[r])
            if r == 0: 
                axes[r][c].set_title(aug_name, fontsize=14)
            axes[r][c].axis('off')

    plt.tight_layout()
    
    # Сохраняем в папку results
    os.makedirs('homework5/results', exist_ok=True)
    path = os.path.join('homework5/results', filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[*] График аугментаций сохранен: {path}")