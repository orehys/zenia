import cv2
import numpy as np
from pathlib import Path

def detect_trees_debug(image_path, output_dir):
    """
    Отладочная версия с сохранением всех промежуточных масок
    """
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Не удалось прочитать: {image_path}")
        return 0
    
    h, w = image.shape[:2]
    print(f"📐 Размер изображения: {w} x {h}")
    
    # Конвертируем в HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # === ПОПРОБУЕМ РАЗНЫЕ ДИАПАЗОНЫ ===
    masks = []
    mask_names = []
    
    # Диапазон 1: Зелёный
    lower1 = np.array([25, 40, 40])
    upper1 = np.array([65, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    masks.append(mask1)
    mask_names.append("green")
    
    # Диапазон 2: Жёлтый
    lower2 = np.array([20, 40, 40])
    upper2 = np.array([35, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    masks.append(mask2)
    mask_names.append("yellow")
    
    # Диапазон 3: Коричневый
    lower3 = np.array([10, 50, 50])
    upper3 = np.array([20, 255, 200])
    mask3 = cv2.inRange(hsv, lower3, upper3)
    masks.append(mask3)
    mask_names.append("brown")
    
    # Диапазон 4: Широкий зелёно-жёлтый
    lower4 = np.array([15, 30, 30])
    upper4 = np.array([75, 255, 255])
    mask4 = cv2.inRange(hsv, lower4, upper4)
    masks.append(mask4)
    mask_names.append("wide_green")
    
    # Сохраняем все маски для отладки
    debug_dir = output_dir / "debug_masks"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (mask, name) in enumerate(zip(masks, mask_names)):
        mask_path = debug_dir / f"{image_path.stem}_{name}.png"
        cv2.imwrite(str(mask_path), mask)
        pixels = cv2.countNonZero(mask)
        percent = 100 * pixels / (w * h)
        print(f"  🎭 Маска {name}: {pixels} пикселей ({percent:.2f}%)")
    
    # Объединяем все маски
    combined_mask = cv2.bitwise_or(masks[0], masks[1])
    combined_mask = cv2.bitwise_or(combined_mask, masks[2])
    combined_mask = cv2.bitwise_or(combined_mask, masks[3])
    
    # Сохраняем объединённую маску
    combined_path = debug_dir / f"{image_path.stem}_combined.png"
    cv2.imwrite(str(combined_path), combined_mask)
    total_pixels = cv2.countNonZero(combined_mask)
    total_percent = 100 * total_pixels / (w * h)
    print(f"  🎭 Объединённая маска: {total_pixels} пикселей ({total_percent:.2f}%)")
    
    # Морфология
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    morph_path = debug_dir / f"{image_path.stem}_morph.png"
    cv2.imwrite(str(morph_path), combined_mask)
    
    # Находим контуры
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n🔍 Найдено контуров: {len(contours)}")
    
    # Пробуем разные пороги площади
    area_thresholds = [500, 1000, 2000, 3000, 5000]
    
    for threshold in area_thresholds:
        tree_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= threshold:
                tree_count += 1
        
        print(f"  🌲 Площадь > {threshold}: {tree_count} деревьев")
    
    # Теперь считаем с оптимальными параметрами
    tree_contours = []
    min_area = 1000  # Попробуем меньший порог
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        # Проверка круглости
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Более широкий диапазон
        if circularity < 0.15 or circularity > 2.5:
            continue
        
        tree_contours.append(contour)
    
    print(f"\n✅ Итого найдено деревьев: {len(tree_contours)}")
    
    # Сохраняем результат
    txt_path = output_dir / f"{image_path.stem}.txt"
    with open(txt_path, 'w') as f:
        for contour in tree_contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 3:
                continue
            
            normalized_points = []
            for point in approx:
                x, y = point[0]
                nx = x / w
                ny = y / h
                normalized_points.extend([nx, ny])
            
            line = f"0 " + " ".join([f"{p:.5f}" for p in normalized_points]) + "\n"
            f.write(line)
    
    # Визуализация
    vis_path = output_dir / f"{image_path.stem}_vis.png"
    vis_image = image.copy()
    cv2.drawContours(vis_image, tree_contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(vis_path), vis_image)
    
    print(f"💾 Разметка: {txt_path.name}")
    print(f"🖼️  Визуализация: {vis_path.name}")
    print(f"📁 Отладочные маски: {debug_dir}/")
    
    return len(tree_contours)


if __name__ == "__main__":
    output_dir = Path("../forest_dataset/labels/train")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ищем изображения
    source_dir = Path(".")
    image_files = list(source_dir.glob("*.JPG")) + list(source_dir.glob("*.jpg"))
    
    if not image_files:
        print("❌ Не найдено изображений!")
        exit(1)
    
    print(f"📁 Найдено изображений: {len(image_files)}")
    print("="*60)
    
    for img_path in image_files:
        print(f"\n🔍 Обработка: {img_path.name}")
        print("="*60)
        detect_trees_debug(img_path, output_dir)
        print("\n")