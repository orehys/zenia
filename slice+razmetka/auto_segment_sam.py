# import os
# os.environ["OMP_NUM_THREADS"] = "8"
import cv2
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import gc
from typing import List, Tuple, Dict

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


def split_image_into_slices(image: np.ndarray, overlap_ratio: float = 0.15) -> List[Dict]:
    """
    Разбивает изображение на 3 вертикальных куска + 2 дополнительных для перекрытия швов.
    
    Args:
        image: Исходное изображение (H x W x C)
        overlap_ratio: Процент перекрытия между кусками (по умолчанию 15%)
    
    Returns:
        Список словарей с информацией о каждом куске:
        - 'image': сам кусок изображения
        - 'x_start': начальная координата X в оригинальном изображении
        - 'x_end': конечная координата X в оригинальном изображении
        - 'slice_index': индекс куска (0-4)
        - 'is_overlap': является ли этот кусок перекрывающим
    """
    h, w = image.shape[:2]
    
    # Основная ширина каждого из 3 кусков
    base_slice_width = w // 3
    
    # Ширина перекрытия
    overlap_width = int(base_slice_width * overlap_ratio)
    
    slices = []
    
    # Создаем 3 основных куска
    for i in range(3):
        x_start = i * base_slice_width
        if i < 2:  # Добавляем перекрытие к первым двум кускам
            x_end = x_start + base_slice_width + overlap_width
        else:  # Последний кусок идет до конца изображения
            x_end = w
        
        # Ограничиваем x_end шириной изображения
        x_end = min(x_end, w)
        
        slice_img = image[:, x_start:x_end].copy()
        
        slices.append({
            'image': slice_img,
            'x_start': x_start,
            'x_end': x_end,
            'slice_index': i,
            'is_overlap': False,
            'original_width': w,
            'original_height': h
        })
    
    # Создаем 2 дополнительных куска для перекрытия швов
    # Первый шов находится между куском 0 и 1
    seam1_center = base_slice_width
    seam1_start = max(0, seam1_center - overlap_width * 2)
    seam1_end = min(w, seam1_center + overlap_width * 2)
    
    seam1_img = image[:, seam1_start:seam1_end].copy()
    slices.append({
        'image': seam1_img,
        'x_start': seam1_start,
        'x_end': seam1_end,
        'slice_index': 3,
        'is_overlap': True,
        'seam_index': 0,
        'original_width': w,
        'original_height': h
    })
    
    # Второй шов находится между куском 1 и 2
    seam2_center = base_slice_width * 2
    seam2_start = max(0, seam2_center - overlap_width * 2)
    seam2_end = min(w, seam2_center + overlap_width * 2)
    
    seam2_img = image[:, seam2_start:seam2_end].copy()
    slices.append({
        'image': seam2_img,
        'x_start': seam2_start,
        'x_end': seam2_end,
        'slice_index': 4,
        'is_overlap': True,
        'seam_index': 1,
        'original_width': w,
        'original_height': h
    })
    
    return slices


def merge_slice_annotations(slices: List[Dict], all_annotations: List[List[np.ndarray]], 
                           original_width: int, original_height: int,
                           blend_ratio: float = 0.3) -> List[np.ndarray]:
    """
    Объединяет аннотации от всех кусков в единый список, убирая дубликаты на швах.
    
    Args:
        slices: Список словарей с информацией о кусках
        all_annotations: Список аннотаций для каждого куска (список контуров)
        original_width: Оригинальная ширина изображения
        original_height: Оригинальная высота изображения
        blend_ratio: Процент от края каждого куска, который удаляется при слиянии
    
    Returns:
        Объединенный список контуров деревьев в координатах оригинального изображения
    """
    merged_contours = []
    
    for idx, (slice_info, contours) in enumerate(zip(slices, all_annotations)):
        x_offset = slice_info['x_start']
        slice_width = slice_info['x_end'] - slice_info['x_start']
        
        # Вычисляем границы обрезки для этого куска
        left_crop = 0
        right_crop = slice_width
        
        if slice_info['is_overlap']:
            # Для перекрывающих кусков - используем центральную часть
            crop_width = int(slice_width * blend_ratio)
            left_crop = crop_width
            right_crop = slice_width - crop_width
        else:
            # Для основных кусков
            if idx == 0:  # Первый кусок - обрезаем только справа
                crop_width = int(slice_width * blend_ratio)
                right_crop = slice_width - crop_width
            elif idx == 1:  # Средний кусок - обрезаем с обеих сторон
                crop_width = int(slice_width * blend_ratio)
                left_crop = crop_width
                right_crop = slice_width - crop_width
            elif idx == 2:  # Последний кусок - обрезаем только слева
                crop_width = int(slice_width * blend_ratio)
                left_crop = crop_width
        
        # Фильтруем и трансформируем контуры
        for contour in contours:
            # Сдвигаем контур в координаты оригинального изображения
            shifted_contour = contour.copy()
            shifted_contour[:, :, 0] += x_offset
            
            # Проверяем, находится ли центр контура в допустимой зоне
            x_coords = shifted_contour[:, :, 0].flatten()
            center_x = np.mean(x_coords)
            
            # Границы допустимой зоны для этого куска
            zone_start = x_offset + left_crop
            zone_end = x_offset + right_crop
            
            # Если центр контура в допустимой зоне - добавляем
            if zone_start <= center_x <= zone_end:
                # Дополнительно проверяем, что контур полностью в пределах изображения
                if (np.all(shifted_contour[:, :, 0] >= 0) and 
                    np.all(shifted_contour[:, :, 0] < original_width) and
                    np.all(shifted_contour[:, :, 1] >= 0) and 
                    np.all(shifted_contour[:, :, 1] < original_height)):
                    merged_contours.append(shifted_contour)
    
    # Удаляем дубликаты (контуры, которые слишком близко друг к другу)
    final_contours = remove_duplicate_contours(merged_contours, threshold=0.7)
    
    return final_contours


def remove_duplicate_contours(contours: List[np.ndarray], threshold: float = 0.7) -> List[np.ndarray]:
    """
    Удаляет дублирующиеся контуры на основе IoU (Intersection over Union).
    
    Args:
        contours: Список контуров
        threshold: Порог IoU для определения дубликатов
    
    Returns:
        Список уникальных контуров
    """
    if len(contours) == 0:
        return []
    
    # Сортируем по площади (большие сначала)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    
    keep_indices = [0]  # Всегда оставляем первый (самый большой)
    
    for i in range(1, len(contours_sorted)):
        is_duplicate = False
        
        for j in keep_indices:
            iou = calculate_contour_iou(contours_sorted[i], contours_sorted[j])
            if iou > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep_indices.append(i)
    
    return [contours_sorted[i] for i in keep_indices]


def calculate_contour_iou(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """
    Вычисляет IoU между двумя контурами.
    """
    # Создаем маски для контуров
    img1 = np.zeros((1000, 1000), dtype=np.uint8)
    img2 = np.zeros((1000, 1000), dtype=np.uint8)
    
    # Нормализуем координаты для масок
    all_points = np.vstack([contour1, contour2])
    min_x = max(0, int(np.min(all_points[:, :, 0])))
    min_y = max(0, int(np.min(all_points[:, :, 1])))
    
    c1_shifted = contour1.copy()
    c2_shifted = contour2.copy()
    c1_shifted[:, :, 0] -= min_x
    c1_shifted[:, :, 1] -= min_y
    c2_shifted[:, :, 0] -= min_x
    c2_shifted[:, :, 1] -= min_y
    
    cv2.drawContours(img1, [c1_shifted], -1, 255, -1)
    cv2.drawContours(img2, [c2_shifted], -1, 255, -1)
    
    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def process_single_slice(slice_img: np.ndarray, mask_generator, forest_mask_slice: np.ndarray) -> List[np.ndarray]:
    """
    Обрабатывает один кусок изображения и возвращает контуры деревьев.
    """
    image_rgb = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(slice_img, cv2.COLOR_BGR2HSV)
    h, w = slice_img.shape[:2]
    
    # Генерация масок SAM
    masks = mask_generator.generate(image_rgb)
    
    tree_contours = []
    
    for mask_data in masks:
        mask = mask_data['segmentation'].astype(np.uint8)
        
        # Проверка пересечения с лесной зоной
        intersection = cv2.bitwise_and(mask, forest_mask_slice)
        intersection_area = cv2.countNonZero(intersection)
        mask_area = cv2.countNonZero(mask)
        
        if mask_area > 0 and intersection_area / mask_area < 0.45:
            continue
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            
            # Динамический порог
            dynamic_min_area = max(3000, (w * h) / 80000)
            
            if is_likely_tree(cnt, image_hsv, forest_mask_slice, min_area=dynamic_min_area):
                tree_contours.append(cnt)
    
    del masks, image_rgb, image_hsv
    gc.collect()
    
    return tree_contours


def process_image_with_slicing(image: np.ndarray, mask_generator, original_width: int, original_height: int) -> List[np.ndarray]:
    """
    Разбивает изображение на куски, обрабатывает каждый кусок отдельно,
    затем объединяет результаты.
    """
    print("  ✂️  Разбиение изображения на куски...")
    slices = split_image_into_slices(image, overlap_ratio=0.15)
    print(f"  📊 Создано кусков: {len(slices)}")
    
    # Выделяем маску леса для всего изображения один раз
    forest_mask_full, forest_contour = create_forest_mask(image)
    forest_area = cv2.contourArea(forest_contour) if forest_contour is not None else 0
    print(f"  ✅ Площадь леса: {forest_area:.0f} пикселей")
    
    all_annotations = []
    
    # Обрабатываем каждый кусок
    for idx, slice_info in enumerate(slices):
        slice_img = slice_info['image']
        x_start = slice_info['x_start']
        x_end = slice_info['x_end']
        
        # Вырезаем соответствующую часть маски леса
        forest_mask_slice = forest_mask_full[:, x_start:x_end].copy()
        
        slice_type = "шов" if slice_info['is_overlap'] else "основной"
        print(f"  🔹 Обработка куска {idx+1}/{len(slices)} ({slice_type}): ширина={slice_img.shape[1]}")
        
        # Обрабатываем кусок
        slice_contours = process_single_slice(slice_img, mask_generator, forest_mask_slice)
        all_annotations.append(slice_contours)
        
        print(f"    🌲 Найдено деревьев: {len(slice_contours)}")
        
        # Очищаем память после каждого куска
        del slice_img, forest_mask_slice, slice_contours
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Объединяем все аннотации
    print("  🔗 Объединение результатов...")
    merged_contours = merge_slice_annotations(slices, all_annotations, original_width, original_height, blend_ratio=0.25)
    
    print(f"  ✅ Всего деревьев после объединения: {len(merged_contours)}")
    
    # Подсчет мелких и крупных
    small_trees = sum(1 for cnt in merged_contours if cv2.contourArea(cnt) < 15000)
    large_trees = len(merged_contours) - small_trees
    print(f"  📊 Из них мелких: {small_trees}, крупных: {large_trees}")
    
    del slices, all_annotations, forest_mask_full
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return merged_contours


def auto_segment_forest(images_dir, output_dir, checkpoint_path="sam_vit_h_4b8939.pth", use_slicing: bool = True):
    """
    Полный пайплайн с улучшенными параметрами и опцией разбиения на куски.
    
    Args:
        images_dir: Путь к директории с изображениями
        output_dir: Путь к директории для сохранения разметки
        checkpoint_path: Путь к файлу модели SAM
        use_slicing: Если True, изображения разбиваются на куски для экономии памяти
    """
    
    # Проверка модели
    if not Path(checkpoint_path).exists():
        print(f"❌ Модель не найдена: {checkpoint_path}")
        return
    
    print("🔄 Загрузка SAM модели...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Устройство: {device}")

    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    
    # === УЛУЧШЕННЫЕ НАСТРОЙКИ SAM ===
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=20,          # Больше точек для мелких деревьев
        pred_iou_thresh=0.84,        # Оптимальный порог
        stability_score_thresh=0.87, # Баланс между точностью и полнотой
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=2500,   # Ловим деревья от 2500 пикселей
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
        
        h, w = image.shape[:2]
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"  📐 Размер: {w} x {h}")
        
        # Проверяем, нужно ли использовать разбиение на куски
        if use_slicing and w > 2000:  # Если ширина больше 2000 пикселей - используем slicing
            print(f"  ✂️  Режим разбиения на куски (use_slicing={use_slicing})")
            tree_contours = process_image_with_slicing(image, mask_generator, w, h)
            forest_mask = None  # В режиме slicing маска леса не создается отдельно
        else:
            print(f"  🔄 Обычный режим обработки (use_slicing={use_slicing} или изображение небольшое)")
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
        if forest_mask is not None and forest_contour is not None:
            forest_overlay = vis_image.copy()
            cv2.drawContours(forest_overlay, [forest_contour], -1, (255, 0, 0), -1)
            cv2.addWeighted(forest_overlay, 0.3, vis_image, 0.7, 0, vis_image)
        
        # Рисуем деревья (зелёные = крупные, жёлтые = мелкие)
        for cnt in tree_contours:
            area = cv2.contourArea(cnt)
            color = (0, 255, 0) if area >= 15000 else (0, 255, 255)
            cv2.drawContours(vis_image, [cnt], -1, color, 2)
        
        cv2.imwrite(str(vis_path), vis_image)
        
        # Сохраняем маску леса (только для обычного режима)
        if forest_mask is not None:
            mask_path = output_path / f"{img_path.stem}_forest_mask.png"
            cv2.imwrite(str(mask_path), forest_mask)
        
        # Очищаем память
        if use_slicing and w > 2000:
            del tree_contours, image, image_rgb, image_hsv
        else:
            del masks, tree_contours, forest_mask, image, image_rgb, image_hsv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        checkpoint_path="sam_vit_h_4b8939.pth",
        use_slicing=True  # Включить режим разбиения на куски для больших изображений
    )