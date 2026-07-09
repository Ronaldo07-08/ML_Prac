import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def plot_comparison(histories, title, filename):
    """Утилита для отрисовки графиков сравнения."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, hist in histories.items():
        ax1.plot(hist['test_accs'], marker='o', label=name)
        ax2.plot(hist['train_losses'], marker='x', linestyle='--', label=f"{name} (Train)")
        ax2.plot(hist['test_losses'], marker='o', label=f"{name} (Test)")
        
    ax1.set_title(f'{title} - Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title(f'{title} - Loss (Train vs Test)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    filepath = os.path.join('homework', 'plots', filename)
    plt.savefig(filepath)
    print(f"[*] График сохранен в: {filepath}")
    plt.show()

def visualize_feature_maps(model, test_loader, device, title, filename):
    """Визуализирует активации первого сверточного слоя"""
    model.eval()
    data, _ = next(iter(test_loader))
    data = data.to(device)
    
    # Ищем первый сверточный слой в модели
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
            
    if first_conv is None: return

    with torch.no_grad():
        activation = F.relu(first_conv(data[0:1]))

    activation = activation.squeeze(0).cpu()
    num_channels = min(16, activation.size(0))

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        if i < num_channels:
            # Отрисовываем тепловую карту того, на что среагировал фильтр
            ax.imshow(activation[i].numpy(), cmap='viridis')
        ax.axis('off')
        
    plt.suptitle(f"Feature Maps (Активации первого слоя): {title}")
    plt.tight_layout()
    filepath = os.path.join('homework', 'plots', filename)
    plt.savefig(filepath)
    print(f"[*] Карта признаков сохранена в: {filepath}")
    plt.show()

