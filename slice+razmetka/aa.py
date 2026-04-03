import cv2
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

def create_forest_mask(image):
    """
    Улучшенное выделение зоны леса (исключает дороги, обочины, воду)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Расширенный диапазон для растительности (зелёный + жёлтый + коричневый)
    lower_veg = np.array([12, 25, 25])
    upper_veg = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_veg, upper_veg)
    
    # Морфологические операции для объединения зон
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Найти самую большую область (основной лесной массив)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Сортируем по площади, берём топ-3
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        forest_mask = np.zeros_like(mask)
        cv2.drawContours(forest_mask, contours, -1, 255, -1)
        return forest_mask, contours[0] if contours else None
    
    return mask, None

def is_likely_tree(contour, image_hsv, forest_mask, min_area=3000, max_area=500000):
    """
    Улучшенная проверка: является ли объект деревом
    """
    area = cv2.contourArea(contour)
    
    # Фильтр по площади
    if area < min_area or area > max_area:
        return False
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    
    # Круглость (деревья обычно 0.3-1.5)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity < 0.25 or circularity > 1.8:
        return False
    
    # Проверка что объект внутри лесной зоны
    x, y, w, h = cv2.boundingRect(contour)
    roi_mask = forest_mask[y:y+h, x:x+w]
    if roi_mask.size > 0:
        forest_ratio = cv2.countNonZero(roi_mask) / roi_mask.size
        if forest_ratio < 0.4:  # Минимум 40% внутри леса
            return False
    
    # Цвет (исключаем серые объекты)
    x, y, w, h = cv2.boundingRect(contour)
    roi = image_hsv[y:y+h, x:x+w]
    roi_mask_contour = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(roi_mask_contour, [contour - [x, y]], -1, 255, -1)
    
    if cv2.countNonZero(roi_mask_contour) > 0:
        mean_color = cv2.mean(roi, mask=roi_mask_contour)[:3]
        s_channel = mean_color[1]
        if s_channel < 18:  # Слишком низкая насыщенность
            return False
    
    return True

def auto_segment_forest(images_dir, output_dir, checkpoint_path="sam_vit_b_01ec64.pth"):
    """
    Полный пайплайн с улучшенными параметрами
    """
    
    # Проверка модели
    if not Path(checkpoint_path).exists():
        print(f"❌ Модель не найдена: {checkpoint_path}")
        return
    
    print("🔄 Загрузка SAM модели...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Устройство: {device}")
    
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    
    # === УЛУЧШЕННЫЕ НАСТРОЙКИ SAM ===
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,          # Уменьшили
        pred_iou_thresh=0.84,
        stability_score_thresh=0.87,
        crop_n_layers=0,             # ⚠️ ВАЖНО: Выключили кроппинг (экономит память!)
        min_mask_region_area=2500,
    )
    
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Поиск изображений
    image_files = (
        list(images_path.glob("*.jpg")) + 
        list(images_path.glob("*.JPG")) + 
        list(images_path.glob("*.png"))
    )
    
    print(f"📁 Найдено изображений: {len(image_files)}")
    print("="*60)
    
    total_trees = 0
    
    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Обработка: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print("  ⚠️  Не удалось прочитать")
            continue
        max_size = 1280  # Максимальная сторона (можно 1024 или 1280)
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  📏 Сжато до: {new_w} x {new_h} (коэффициент: {scale:.2f})")
        h, w = image.shape[:2]
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"  📐 Размер: {w} x {h}")
        
        # ЭТАП 1: Выделение зоны леса
        print("  🌲 Выделение зоны леса...")
        forest_mask, forest_contour = create_forest_mask(image)
        forest_area = cv2.contourArea(forest_contour) if forest_contour is not None else 0
        print(f"  ✅ Площадь леса: {forest_area:.0f} пикселей")
        
        # ЭТАП 2: Генерация масок SAM
        print("  🔄 Генерация масок SAM...")
        masks = mask_generator.generate(image_rgb)
        print(f"  📊 Найдено объектов SAM: {len(masks)}")
        
        # ЭТАП 3: Фильтрация
        print("  🔍 Фильтрация (лес + форма + цвет)...")
        tree_contours = []
        small_trees = 0
        large_trees = 0
        
        for mask_data in masks:
            mask = mask_data['segmentation'].astype(np.uint8)
            
            # Проверка пересечения с лесной зоной
            intersection = cv2.bitwise_and(mask, forest_mask)
            intersection_area = cv2.countNonZero(intersection)
            mask_area = cv2.countNonZero(mask)
            
            if mask_area > 0 and intersection_area / mask_area < 0.45:
                continue
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                
                # Динамический порог в зависимости от размера изображения
                dynamic_min_area = max(3000, (w * h) / 80000)
                
                if is_likely_tree(cnt, image_hsv, forest_mask, min_area=dynamic_min_area):
                    tree_contours.append(cnt)
                    if area < 15000:
                        small_trees += 1
                    else:
                        large_trees += 1
        
        print(f"  ✅ Найдено деревьев: {len(tree_contours)} (мелких: {small_trees}, крупных: {large_trees})")
        total_trees += len(tree_contours)
        
        # ЭТАП 4: Сохранение в YOLO
        txt_path = output_path / f"{img_path.stem}.txt"
        with open(txt_path, 'w') as f:
            for cnt in tree_contours:
                epsilon = 0.001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
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
        
        # ЭТАП 5: Визуализация
        vis_path = output_path / f"{img_path.stem}_vis.jpg"
        vis_image = image.copy()
        
        # Рисуем зону леса (полупрозрачный синий)
        if forest_contour is not None:
            forest_overlay = vis_image.copy()
            cv2.drawContours(forest_overlay, [forest_contour], -1, (255, 0, 0), -1)
            cv2.addWeighted(forest_overlay, 0.3, vis_image, 0.7, 0, vis_image)
        
        # Рисуем деревья (зелёные = крупные, жёлтые = мелкие)
        for cnt in tree_contours:
            area = cv2.contourArea(cnt)
            color = (0, 255, 0) if area >= 15000 else (0, 255, 255)
            cv2.drawContours(vis_image, [cnt], -1, color, 2)
        
        cv2.imwrite(str(vis_path), vis_image)
        
        # Сохраняем маску леса
        mask_path = output_path / f"{img_path.stem}_forest_mask.png"
        cv2.imwrite(str(mask_path), forest_mask)
        
        print(f"  💾 Разметка: {txt_path.name}")
        print(f"  🖼️  Визуализация: {vis_path.name}")
        print(f"  🎨 Зелёные = крупные, Жёлтые = мелкие деревья")
    
    print("\n" + "="*60)
    print(f"🎉 ГОТОВО!")
    print(f"📊 Всего изображений: {len(image_files)}")
    print(f"🌲 Всего деревьев: {total_trees}")
    print(f"📁 Разметка сохранена в: {output_path}")
    print("="*60)

if __name__ == "__main__":
    auto_segment_forest(
        images_dir="../forest_dataset/images/train",
        output_dir="../forest_dataset/labels/train",
        checkpoint_path="sam_vit_b_01ec64.pth"  # ← новое имя
    )