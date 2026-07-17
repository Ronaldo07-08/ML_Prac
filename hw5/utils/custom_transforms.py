import random
from PIL import Image, ImageEnhance, ImageFilter

class CustomRandomBlur:
    """Кастомное случайное размытие (Blur)."""
    def __init__(self, p=0.5, radius=(1, 3)):
        self.p = p
        self.radius = radius
        
    def __call__(self, img):
        if random.random() < self.p:
            r = random.uniform(*self.radius)
            return img.filter(ImageFilter.GaussianBlur(r))
        return img

class CustomColorJitter:
    """Кастомное изменение яркости и контраста."""
    def __init__(self, p=0.5, brightness=1.5, contrast=1.5):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        
    def __call__(self, img):
        if random.random() < self.p:
            # Вычисляем случайный фактор: 1.0 - это оригинал
            b_factor = random.uniform(max(0, 1 - self.brightness/2), 1 + self.brightness/2)
            c_factor = random.uniform(max(0, 1 - self.contrast/2), 1 + self.contrast/2)
            img = ImageEnhance.Brightness(img).enhance(b_factor)
            img = ImageEnhance.Contrast(img).enhance(c_factor)
        return img

class CustomPixelate:
    """Кастомная пикселизация (уменьшение и увеличение размера без сглаживания)."""
    def __init__(self, p=0.5, pixel_size=4):
        self.p = p
        self.pixel_size = pixel_size
        
    def __call__(self, img):
        if random.random() < self.p:
            w, h = img.size
            # Уменьшаем (сжимаем)
            img_small = img.resize((w // self.pixel_size, h // self.pixel_size), Image.NEAREST)
            # Растягиваем обратно без сглаживания
            img = img_small.resize((w, h), Image.NEAREST)
        return img