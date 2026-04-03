from pathlib import Path

def check_dataset():
    dataset_path = Path('C:/Users/Sasha/Desktop/Zenia/kursach/forest_dataset')
    
    print("🔍 Проверка структуры датасета...\n")
    
    # Проверка data.yaml
    yaml_path = dataset_path / 'data.yaml'
    if yaml_path.exists():
        print("✅ data.yaml найден")
        # Просто читаем текст, не парсим
        content = yaml_path.read_text()
        print(f"   Содержимое:\n{content}")
    else:
        print("❌ data.yaml не найден!")
    
    print("\n" + "="*60 + "\n")
    
    # Проверка папок
    train_images = dataset_path / 'images' / 'train'
    train_labels = dataset_path / 'labels' / 'train'
    val_images = dataset_path / 'images' / 'val'
    val_labels = dataset_path / 'labels' / 'val'
    
    print("📁 Папки:")
    print(f"   Train images: {train_images.exists()} ({len(list(train_images.glob('*.JPG')) + list(train_images.glob('*.jpg'))) if train_images.exists() else 0} файлов)")
    print(f"   Train labels: {train_labels.exists()} ({len(list(train_labels.glob('*.txt'))) if train_labels.exists() else 0} файлов)")
    print(f"   Val images: {val_images.exists()} ({len(list(val_images.glob('*.JPG')) + list(val_images.glob('*.jpg'))) if val_images.exists() else 0} файлов)")
    print(f"   Val labels: {val_labels.exists()} ({len(list(val_labels.glob('*.txt'))) if val_labels.exists() else 0} файлов)")
    
    print("\n" + "="*60 + "\n")
    
    # Проверка соответствия имен
    if train_images.exists() and train_labels.exists():
        # Ищем и JPG и jpg
        img_files = list(train_images.glob('*.JPG')) + list(train_images.glob('*.jpg'))
        lbl_files = list(train_labels.glob('*.txt'))
        
        img_names = set([f.stem for f in img_files])
        lbl_names = set([f.stem for f in lbl_files])
        
        missing_labels = img_names - lbl_names
        missing_images = lbl_names - img_names
        
        if missing_labels:
            print(f"⚠️  Нет разметки для {len(missing_labels)} изображений")
            print(f"   Примеры: {list(missing_labels)[:5]}")
        if missing_images:
            print(f"⚠️  Нет изображений для {len(missing_images)} разметок")
            print(f"   Примеры: {list(missing_images)[:5]}")
        
        if not missing_labels and not missing_images:
            print("✅ Все изображения имеют разметку!")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    check_dataset()