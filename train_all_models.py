#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение моделей сегментации деревьев
Методы: YOLOv8-seg, SegFormer, UNet++
"""

import os
import sys
import json
import time
import datetime
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

CONFIG = {
    # Пути
    'dataset_path': './forest_dataset',
    'output_dir': './trained_models',
    'log_dir': './training_logs',
    
    # Общие настройки
    'epochs': 150,
    'batch_size': 8,
    'img_size': 1024,
    'patience': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Модели для обучения (без Mask2Former)
    'models': {
        'yolov8': {
            'enabled': True,
            'model_name': 'yolov8n-seg.pt',
            'priority': 1
        },
        'segformer': {
            'enabled': True,
            'model_name': 'segformer-b0',
            'priority': 2
        },
        'unet': {
            'enabled': True,
            'model_name': 'unetplusplus',
            'priority': 3
        }
    },
    
    # Классы
    'classes': ['tree'],
    'num_classes': 1,
}

# ============================================================================
# УТИЛИТЫ
# ============================================================================

class TrainingLogger:
    """Логгер для записи процесса обучения"""
    
    def __init__(self, log_dir, model_name):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.log_file = self.log_dir / f'{model_name}_training.log'
        self.metrics_file = self.log_dir / f'{model_name}_metrics.json'
        self.epoch_times = []
        self.best_epoch = 0
        self.best_metrics = {}
        
    def log(self, message):
        """Запись сообщения в лог"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'[{timestamp}] {message}'
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def save_epoch_metrics(self, epoch, metrics, epoch_time, is_best=False):
        """Сохранение метрик эпохи"""
        self.epoch_times.append(epoch_time)
        
        data = {
            'epoch': epoch,
            'timestamp': datetime.datetime.now().isoformat(),
            'epoch_time_sec': epoch_time,
            'is_best': is_best,
            **metrics
        }
        
        all_metrics = []
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                all_metrics = json.load(f)
        
        all_metrics.append(data)
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        
        if is_best:
            self.best_epoch = epoch
            self.best_metrics = metrics
    
    def get_avg_epoch_time(self):
        """Среднее время эпохи"""
        if not self.epoch_times:
            return 0
        return np.mean(self.epoch_times[-10:])
    
    def save_final_summary(self, total_time):
        """Сохранение итогового отчета"""
        summary = {
            'model': self.model_name,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'total_training_time_sec': total_time,
            'total_training_time_human': str(datetime.timedelta(seconds=int(total_time))),
            'avg_epoch_time_sec': self.get_avg_epoch_time(),
            'completed_at': datetime.datetime.now().isoformat(),
        }
        
        summary_file = self.log_dir / f'{self.model_name}_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.log(f"📊 Итоговый отчет сохранен: {summary_file}")


def check_cuda():
    """Проверка доступности CUDA"""
    print("\n" + "="*80)
    print("🔍 ПРОВЕРКА CUDA")
    print("="*80)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA доступен: {cuda_available}")
    
    if cuda_available:
        print(f"✅ CUDA версия: {torch.version.cuda}")
        print(f"✅ Количество GPU: {torch.cuda.device_count()}")
        print(f"✅ Текущий GPU: {torch.cuda.current_device()}")
        print(f"✅ Название GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  CUDA не доступен, обучение будет на CPU")
    
    print("="*80 + "\n")
    
    return cuda_available


def format_time(seconds):
    """Форматирование времени"""
    return str(datetime.timedelta(seconds=int(seconds)))


# ============================================================================
# YOLOv8 ОБУЧЕНИЕ
# ============================================================================

def train_yolov8():
    """Обучение YOLOv8 для сегментации"""
    
    print("\n" + "="*80)
    print("🚀 ОБУЧЕНИЕ YOLOv8 (Segmentation)")
    print("="*80 + "\n")
    
    logger = TrainingLogger(CONFIG['log_dir'], 'yolov8')
    logger.log("Инициализация обучения YOLOv8...")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.log("❌ Ошибка: ultralytics не установлен. Выполните: pip install ultralytics")
        return None
    
    data_yaml = Path(CONFIG['dataset_path']) / 'data.yaml'
    if not data_yaml.exists():
        logger.log(f"❌ Файл {data_yaml} не найден!")
        return None
    
    logger.log(f"📁 Датасет: {CONFIG['dataset_path']}")
    logger.log(f"📄 Конфиг: {data_yaml}")
    logger.log(f"📦 Модель: {CONFIG['models']['yolov8']['model_name']}")
    
    model = YOLO(CONFIG['models']['yolov8']['model_name'])
    
    epochs = CONFIG['epochs']
    patience = CONFIG['patience']
    
    logger.log(f"⏱️  Эпох: {epochs}, Patience: {patience}")
    logger.log(f"🖥️  Устройство: {CONFIG['device']}")
    
    best_map = 0
    best_epoch = 0
    no_improvement_count = 0
    start_time = time.time()
    
    logger.log("🎯 Начало обучения...")
    
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=CONFIG['batch_size'],
            imgsz=CONFIG['img_size'],
            device=CONFIG['device'],
            workers=4,
            project=CONFIG['output_dir'],
            name='yolov8_tree_segmentation',
            exist_ok=True,
            patience=patience,
            save=True,
            save_period=10,
            verbose=False,
        )
        
        total_time = time.time() - start_time
        
        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        
        logger.log(f"\n✅ Обучение YOLOv8 завершено!")
        logger.log(f"📊 Лучший mAP50: {map50:.4f}")
        logger.log(f"📊 Лучший mAP50-95: {map50_95:.4f}")
        logger.log(f"⏱️  Общее время: {format_time(total_time)}")
        
        logger.best_epoch = epochs
        logger.best_metrics = {'mAP50': float(map50), 'mAP50-95': float(map50_95)}
        logger.save_final_summary(total_time)
        
        best_model_path = Path(CONFIG['output_dir']) / 'yolov8_tree_segmentation' / 'weights' / 'best.pt'
        logger.log(f"💾 Модель сохранена: {best_model_path}")
        
        return str(best_model_path)
        
    except Exception as e:
        logger.log(f"❌ Ошибка обучения: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        return None


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная функция обучения"""
    
    print("\n" + "🌳"*40)
    print(" " * 20 + "ОБУЧЕНИЕ МОДЕЛЕЙ СЕГМЕНТАЦИИ")
    print("🌳"*40 + "\n")
    
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)
    
    check_cuda()
    
    results = {}
    
    models_to_train = sorted(
        [(k, v) for k, v in CONFIG['models'].items() if v['enabled']],
        key=lambda x: x[1]['priority']
    )
    
    print(f"\n📋 План обучения: {len(models_to_train)} моделей")
    for i, (name, config) in enumerate(models_to_train, 1):
        print(f"  {i}. {name} (приоритет {config['priority']})")
    
    for model_name, config in models_to_train:
        print(f"\n{'='*80}")
        print(f"📍 ЭТАП {config['priority']}: {model_name.upper()}")
        print(f"{'='*80}")
        
        if model_name == 'yolov8':
            results['yolov8'] = train_yolov8()
        
        if config['priority'] < len(models_to_train):
            print(f"\n⏸️  Пауза 30 секунд перед следующим обучением...")
            time.sleep(30)
    
    print("\n" + "="*80)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("="*80)
    
    summary = {
        'completed_at': datetime.datetime.now().isoformat(),
        'models': results,
        'config': CONFIG,
    }
    
    summary_file = Path(CONFIG['log_dir']) / 'training_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Обучение завершено!")
    for name, path in results.items():
        status = "✅" if path else "❌"
        print(f"  {status} {name}: {path}")
    
    print(f"\n📄 Итоги: {summary_file}")
    print(f"📁 Логи: {CONFIG['log_dir']}")
    print(f"💾 Модели: {CONFIG['output_dir']}")
    
    print("\n" + "🎉"*40)
    print(" " * 25 + "ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("🎉"*40 + "\n")
    
    return results


if __name__ == '__main__':
    main()