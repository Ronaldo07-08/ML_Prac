import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os

from generator_model import GeneratorTransformer
from text_dataset import get_text_dataloader

def train_generator():
    print("=== Обучение текстового генератора ===")
    
    config = {
        'batch_size': 4,
        'max_length': 128,
        'learning_rate': 3e-4,
        'num_epochs': 10,
        'd_model': 256,
        'num_heads': 8,
        'd_ff': 512,
        'num_layers': 4,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Используем устройство: {device}")
    

    text_file = "transformer_basics/training_text.txt" 
    tokenizer_file = "transformer_basics/mistral_tokenizer.json"
    
    if not os.path.exists(text_file):
        print(f"[!] ОШИБКА: Файл {text_file} не найден.")
        print("Пожалуйста, создайте текстовый файл (например, скачайте книгу в формате .txt) и положите по этому пути.")
        return

    train_loader, tokenizer = get_text_dataloader(text_file, tokenizer_file, config['batch_size'], config['max_length'])
    vocab_size = tokenizer.get_vocab_size()
    
    model = GeneratorTransformer(
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_len=config['max_length'],
        device=device
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Инструменты для Mixed Precision
    scaler = GradScaler()
    
    # Цикл обучения
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for input_ids, target_ids in pbar:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            optimizer.zero_grad()
            
            # --- Mixed Precision Forward ---
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                logits = model(input_ids) # [batch, seq_len, vocab_size]
                
                # Переформатируем тензоры для CrossEntropyLoss
                # logits: [batch * seq_len, vocab_size]
                # targets: [batch * seq_len]
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            
            # --- Mixed Precision Backward ---
            scaler.scale(loss).backward()
            
            # Защита от взрыва градиентов
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"[*] Эпоха {epoch+1} завершена. Средний Loss: {avg_loss:.4f}")
        
        # Тестовая генерация после каждой эпохи, чтобы видеть прогресс
        print(f"--- Тестовая генерация ---")
        prompt = "The "
        sample = model.generate(prompt, temperature=0.8, max_out_tokens=50)
        print(f"Prompt: '{prompt}' -> {sample}\n")

    # Сохранение чекпоинта
    os.makedirs('transformer_basics/checkpoints', exist_ok=True)
    save_path = 'transformer_basics/checkpoints/generator_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, save_path)
    print(f"[*] Модель сохранена в {save_path}")

def chat():
    print("=== Чат с генератором текста ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer_file = "transformer_basics/mistral_tokenizer.json"
    checkpoint_path = "transformer_basics/checkpoints/generator_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print("[!] ОШИБКА: Чекпоинт модели не найден. Сначала запустите обучение!")
        return
        
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_file)
    
    model = GeneratorTransformer.load_from_checkpoint(checkpoint_path, tokenizer, str(device))
    print("[*] Модель успешно загружена! Введите 'quit' для выхода.\n")
    
    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ['quit', 'exit', 'выход']:
            break
            
        if not user_input.strip():
            continue
            
        response = model.generate(
            prompt=user_input, 
            context_len=128, 
            temperature=0.7, 
            max_out_tokens=100
        )
        print(f"Бот: {response}\n")

if __name__ == "__main__":
    # 1. Сначала нужно обучить модель 
    # train_generator()
    
    # 2. После обучения закомментировать train_generator() и раскомментировать chat()
    chat()