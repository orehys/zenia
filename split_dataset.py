import shutil
from pathlib import Path
import random

def split_dataset(dataset_path, val_ratio=0.2):
    """Разделение датасета на train/val (80/20)"""
    
    dataset = Path(dataset_path)
    
    train_images = dataset / 'images' / 'train'
    train_labels = dataset / 'labels' / 'train'
    val_images = dataset / 'images' / 'val'
    val_labels = dataset / 'labels' / 'val'
    
    # Создаём папки val
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    
    # Получаем все изображения
    images = list(train_images.glob('*.JPG')) + list(train_images.glob('*.jpg'))
    random.shuffle(images)
    
    val_count = max(1, int(len(images) * val_ratio))  # Минимум 1 файл
    val_images_list = images[:val_count]
    
    print(f"📊 Всего изображений: {len(images)}")
    print(f"📁 Train: {len(images) - val_count}")
    print(f"📁 Val: {val_count}")
    
    # Перемещаем файлы
    for img_path in val_images_list:
        name = img_path.stem
        
        # Изображение
        shutil.move(str(img_path), str(val_images / img_path.name))
        
        # Разметка
        label_src = train_labels / f"{name}.txt"
        if label_src.exists():
            shutil.move(str(label_src), str(val_labels / f"{name}.txt"))
    
    print("✅ Готово!")

if __name__ == '__main__':
    split_dataset('/Users/golcov/prog/kursach/forest_dataset', val_ratio=0.2)