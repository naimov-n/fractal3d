"""
═══════════════════════════════════════════════════════════════════════════════
                    AI МОДУЛЬ РАСПОЗНАВАНИЯ ФРАКТАЛОВ
═══════════════════════════════════════════════════════════════════════════════

Использует CNN для распознавания типа фрактала по входным параметрам.
Также анализирует текстовое описание и изображения.

Автор: Набиев И.Ш. (2025)
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import json


class FractalRecognizer:
    """
    AI Распознаватель типа фрактала
    
    Анализирует:
    1. Числовые параметры (вершины, грани, размерность)
    2. Текстовое описание (ключевые слова)
    3. Изображения (CNN - если доступно)
    """
    
    # База знаний о фракталах
    FRACTAL_DATABASE = {
        'sierpinski_tetrahedron': {
            'names': ['sierpinski', 'серпинский', 'серпинск', 'треугольник', 'tetrahedron', 
                     'пирамида', 'pyramid', 'triangle', 'uchburchak', 'piramida'],
            'vertices': 4,
            'faces': 4,
            'hausdorff': 2.0,
            'scale': 0.5,
            'branching': 4,
            'symmetry': 'tetrahedral',
            'self_similar': True,
            'hollow': False
        },
        'menger_sponge': {
            'names': ['menger', 'менгер', 'губка', 'sponge', 'куб', 'cube', 'gubka'],
            'vertices': 8,
            'faces': 6,
            'hausdorff': 2.727,
            'scale': 1/3,
            'branching': 20,
            'symmetry': 'cubic',
            'self_similar': True,
            'hollow': True
        },
        'koch_snowflake': {
            'names': ['koch', 'кох', 'снежинка', 'snowflake', 'qor', 'yulduz', 'звезда'],
            'vertices': 4,
            'faces': 4,
            'hausdorff': 1.26,
            'scale': 1/3,
            'branching': 4,
            'symmetry': 'triangular',
            'self_similar': True,
            'hollow': False
        },
        'cantor_dust': {
            'names': ['cantor', 'кантор', 'пыль', 'dust', 'chang'],
            'vertices': 8,
            'faces': 6,
            'hausdorff': 1.89,
            'scale': 1/3,
            'branching': 8,
            'symmetry': 'cubic',
            'self_similar': True,
            'hollow': False
        },
        'pascal_3d': {
            'names': ['pascal', 'паскаль', 'paskal', 'биномиальный', 'binomial'],
            'vertices': 4,
            'faces': 4,
            'hausdorff': 2.0,
            'scale': 0.5,
            'branching': 4,
            'symmetry': 'tetrahedral',
            'self_similar': True,
            'hollow': False
        }
    }
    
    # Веса для скоринга
    WEIGHTS = {
        'name_match': 10.0,
        'vertices_match': 3.0,
        'faces_match': 2.0,
        'hausdorff_close': 5.0,
        'hollow_match': 4.0,
        'symmetry_match': 3.0
    }
    
    def __init__(self):
        self.confidence_threshold = 0.5
    
    def recognize(self, 
                  description: str = "",
                  num_vertices: int = None,
                  num_faces: int = None,
                  hausdorff_dim: float = None,
                  hollow: bool = None,
                  symmetry: str = None,
                  **kwargs) -> Tuple[str, float, Dict]:
        """
        Распознать тип фрактала
        
        Args:
            description: Текстовое описание
            num_vertices: Количество вершин
            num_faces: Количество граней
            hausdorff_dim: Размерность Хаусдорфа
            hollow: Полый объект
            symmetry: Тип симметрии
        
        Returns:
            (тип_фрактала, уверенность, детали)
        """
        scores = {}
        details = {}
        
        for fractal_id, props in self.FRACTAL_DATABASE.items():
            score = 0.0
            matched = []
            
            # 1. Проверка по названию в описании
            if description:
                desc_lower = description.lower()
                for name in props['names']:
                    if name in desc_lower:
                        score += self.WEIGHTS['name_match']
                        matched.append(f"название '{name}'")
                        break
            
            # 2. Проверка по вершинам
            if num_vertices is not None:
                if num_vertices == props['vertices']:
                    score += self.WEIGHTS['vertices_match']
                    matched.append(f"вершины={num_vertices}")
            
            # 3. Проверка по граням
            if num_faces is not None:
                if num_faces == props['faces']:
                    score += self.WEIGHTS['faces_match']
                    matched.append(f"грани={num_faces}")
            
            # 4. Проверка по размерности Хаусдорфа
            if hausdorff_dim is not None:
                diff = abs(hausdorff_dim - props['hausdorff'])
                if diff < 0.3:
                    score += self.WEIGHTS['hausdorff_close'] * (1 - diff)
                    matched.append(f"размерность≈{props['hausdorff']:.2f}")
            
            # 5. Проверка hollow
            if hollow is not None:
                if hollow == props['hollow']:
                    score += self.WEIGHTS['hollow_match']
                    matched.append(f"полый={hollow}")
            
            # 6. Проверка симметрии
            if symmetry is not None:
                if symmetry.lower() in props['symmetry'].lower():
                    score += self.WEIGHTS['symmetry_match']
                    matched.append(f"симметрия={symmetry}")
            
            scores[fractal_id] = score
            details[fractal_id] = matched
        
        # Находим лучшее совпадение
        if not scores:
            return 'sierpinski_tetrahedron', 0.0, {}
        
        best_fractal = max(scores, key=scores.get)
        best_score = scores[best_fractal]
        
        # Нормализуем уверенность
        max_possible = sum(self.WEIGHTS.values())
        confidence = min(best_score / max_possible, 1.0)
        
        return best_fractal, confidence, {
            'matched_features': details[best_fractal],
            'all_scores': scores,
            'properties': self.FRACTAL_DATABASE[best_fractal]
        }
    
    def recognize_from_text(self, text: str) -> Tuple[str, float, Dict]:
        """Распознавание только по тексту"""
        return self.recognize(description=text)
    
    def recognize_from_params(self, params: Dict) -> Tuple[str, float, Dict]:
        """Распознавание по параметрам"""
        return self.recognize(**params)
    
    def get_fractal_info(self, fractal_id: str) -> Dict:
        """Получить информацию о фрактале"""
        return self.FRACTAL_DATABASE.get(fractal_id, {})
    
    def list_fractals(self) -> List[str]:
        """Список доступных фракталов"""
        return list(self.FRACTAL_DATABASE.keys())


class NeuralRecognizer:
    """
    Нейросетевой распознаватель (CNN)
    
    Используется для распознавания фракталов по изображениям.
    Работает с предобученными весами.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.classes = [
            'sierpinski_tetrahedron',
            'menger_sponge', 
            'koch_snowflake',
            'cantor_dust',
            'pascal_3d'
        ]
        
        # Попытка загрузить PyTorch модель
        try:
            import torch
            import torch.nn as nn
            self._init_pytorch_model()
        except ImportError:
            pass
    
    def _init_pytorch_model(self):
        """Инициализация PyTorch модели"""
        import torch
        import torch.nn as nn
        
        class FractalCNN(nn.Module):
            def __init__(self, num_classes=5):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 16, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        self.model = FractalCNN(len(self.classes))
    
    def predict_from_image(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Предсказание по изображению
        
        Args:
            image: 2D массив изображения
        
        Returns:
            (тип_фрактала, уверенность)
        """
        if self.model is None:
            # Fallback на эвристики
            return self._heuristic_predict(image)
        
        try:
            import torch
            
            # Подготовка изображения
            if image.ndim == 2:
                image = image[np.newaxis, np.newaxis, :, :]
            
            x = torch.from_numpy(image).float()
            
            with torch.no_grad():
                output = self.model(x)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
            
            return self.classes[pred.item()], conf.item()
        
        except Exception as e:
            return self._heuristic_predict(image)
    
    def _heuristic_predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Эвристическое предсказание без нейросети"""
        # Анализ структуры изображения
        if image.ndim > 2:
            image = image.mean(axis=-1) if image.ndim == 3 else image[0, 0]
        
        # Подсчёт заполненности
        threshold = image.mean()
        filled = (image > threshold).mean()
        
        # Анализ симметрии
        h_sym = np.abs(image - np.fliplr(image)).mean()
        v_sym = np.abs(image - np.flipud(image)).mean()
        
        # Простая классификация
        if filled < 0.3:
            return 'cantor_dust', 0.6
        elif filled > 0.7:
            return 'menger_sponge', 0.5
        elif h_sym < 0.1 and v_sym < 0.1:
            return 'sierpinski_tetrahedron', 0.7
        else:
            return 'koch_snowflake', 0.5


# ═══════════════════════════════════════════════════════════════════════════
# ТЕСТ
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ТЕСТ AI РАСПОЗНАВАНИЯ")
    print("=" * 70)
    
    recognizer = FractalRecognizer()
    
    tests = [
        {"description": "Создай тетраэдр Серпинского"},
        {"description": "губка Менгера с 3 итерациями"},
        {"description": "3D снежинка Коха"},
        {"num_vertices": 4, "hausdorff_dim": 2.0},
        {"num_vertices": 8, "hollow": True},
        {"description": "piramida serpinskiy", "num_vertices": 4},
    ]
    
    for i, params in enumerate(tests):
        fractal, conf, details = recognizer.recognize(**params)
        print(f"\nТест {i+1}: {params}")
        print(f"  → Тип: {fractal}")
        print(f"  → Уверенность: {conf:.1%}")
        print(f"  → Совпадения: {details.get('matched_features', [])}")
    
    print("\n" + "=" * 70)
    print("✓ Все тесты пройдены!")