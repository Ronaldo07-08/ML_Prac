import torch
import torch.nn as nn
import os
import sys

from layers import MultiheadAttention, FeedForward, Embedding

def get_subsequent_mask(seq_len):
    """Создает треугольную маску, чтобы модель не заглядывала в будущее"""
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask

class DecoderOnlyLayer(nn.Module):
    """Слой декодера БЕЗ cross-attention (только для генерации текста)"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1. Masked Self-Attention
        x_norm = self.layernorm1(x)
        attn_out, _ = self.self_attention(x_norm, x_norm, x_norm, mask)
        x = x + attn_out
        
        # 2. Feed-Forward
        x_norm = self.layernorm2(x)
        ffn_out = self.ffn(x_norm)
        x = self.dropout(x + ffn_out)
        return x

class GeneratorTransformer(nn.Module):
    """Decoder-only Transformer (подобный GPT) для генерации текста"""
    def __init__(
            self, 
            vocab_size: int, 
            tokenizer,
            d_model: int = 256, 
            num_heads: int = 8, 
            d_ff: int = 512, 
            num_layers: int = 4, 
            pad_index: int = 0, 
            dropout: float = 0.1,
            max_len: int = 192,
            device: str = 'cpu'
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.max_len = max_len
        self.tokenizer = tokenizer
        
        # Токены (для Mistral tokenizer: eos = 2, pad = 0 (unk))
        self.pad_index = pad_index
        self.eos_token_id = 2 
        
        self.embedding = Embedding(d_model, vocab_size, pad_index)
        
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.normalize = nn.LayerNorm(d_model)
        self.vocab_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        # Создаем маску для авторегрессии (чтобы токен i не видел токен i+1)
        mask = get_subsequent_mask(seq_len).unsqueeze(0).to(self.device)
        
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.normalize(x)
        logits = self.vocab_projection(x)
        return logits
    
    def generate(self, prompt: str, context_len: int = 128, temperature: float = 0.8, max_out_tokens: int = 100):
        """Авторегрессивная генерация текста со сдвигом контекста"""
        self.eval()
        with torch.no_grad():
            #  Токенизация промпта
            input_ids = self.tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            generated = input_ids.clone()
            
            for _ in range(max_out_tokens):
                #  Ограничиваем длину контекста, чтобы не превысить max_len
                current_context = generated[:, -context_len:]
                
                #  Предсказание
                outputs = self(current_context)
                
                #  Берем логиты только для последнего токена в последовательности
                next_token_logits = outputs[0, -1, :] / temperature
                
                #  Сэмплирование
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)
                
                #  Добавление к сгенерированному тексту
                generated = torch.cat([generated, next_token], dim=1)
                
                #  Проверка на конец генерации
                if next_token.item() == self.eos_token_id:
                    break
                    
        return self.tokenizer.decode(generated[0].tolist())

    def generate_beam_search(self, prompt: str, beam_size: int = 3, max_out_tokens: int = 50, temperature: float = 0.8):
        """Реализация Beam Search с исправленным штрафом за повторение."""
        self.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            # Расширяем входные данные для всех лучей
            sequences = input_ids.repeat(beam_size, 1)
            scores = torch.zeros(beam_size).to(self.device)
            
            for _ in range(max_out_tokens):
                # Forward pass
                outputs = self(sequences) # [beam_size, seq_len, vocab_size]
                
                # Берем логиты последнего токена для всех лучей сразу
                next_token_logits = outputs[:, -1, :] / temperature
                
                # --- ИСПРАВЛЕННЫЙ ШТРАФ ЗА ПОВТОРЕНИЕ ---
                # Применяем штраф отдельно для каждого луча
                penalty = 1.2
                for i in range(beam_size):
                    # Берем токены, которые уже есть в текущем луче i
                    used_tokens = set(sequences[i].tolist())
                    for token in used_tokens:
                        next_token_logits[i, token] /= penalty

                # Считаем log_softmax
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Кумулятивные скоры
                candidate_scores = scores.unsqueeze(1) + log_probs
                
                # Ищем лучшие варианты
                flat_scores = candidate_scores.view(-1)
                top_scores, top_indices = torch.topk(flat_scores, beam_size)
                
                # Восстанавливаем индексы
                beam_indices = top_indices // self.vocab_size
                token_indices = top_indices % self.vocab_size
                
                # Обновляем sequences и scores
                sequences = torch.cat([sequences[beam_indices], token_indices.unsqueeze(1)], dim=1)
                scores = top_scores
                
            # Возвращаем лучшую последовательность
            best_beam = scores.argmax().item()
            return self.tokenizer.decode(sequences[best_beam].tolist())

    @classmethod
    def load_from_checkpoint(cls, path: str, tokenizer, device: str):
        """Вспомогательный метод для загрузки модели"""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config = checkpoint['config']
        
        model = cls(
            vocab_size=tokenizer.get_vocab_size(),
            tokenizer=tokenizer,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            max_len=config['max_length'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model