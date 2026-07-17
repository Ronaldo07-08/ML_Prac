import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import os

class SlidingWindowTextDataset(Dataset):
    """Датасет, который нарезает длинный текст на окна для обучения авторегрессии"""
    def __init__(self, text_path: str, tokenizer: Tokenizer, max_length: int = 128, stride: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Шаг сдвига окна
        self.stride = stride if stride is not None else max_length // 2
        
        print(f"[*] Чтение и токенизация файла: {text_path}")
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Токенизируем весь текст целиком
        self.full_tokens = self.tokenizer.encode(text).ids
        print(f"[*] Всего токенов в тексте: {len(self.full_tokens):,}")
        
        # Создаем индексы для окон
        self.windows = []
        # Мы должны оставить +1 токен для таргета
        for i in range(0, len(self.full_tokens) - max_length, self.stride):
            self.windows.append(i)
            
        print(f"[*] Создано {len(self.windows):,} блоков контекста")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start_idx = self.windows[idx]
        
        # Берем блок длиной max_length + 1
        chunk = self.full_tokens[start_idx : start_idx + self.max_length + 1]
        
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        
        return input_ids, target_ids

def get_text_dataloader(text_path: str, tokenizer_path: str, batch_size: int = 4, max_length: int = 128):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    dataset = SlidingWindowTextDataset(text_path, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return loader, tokenizer