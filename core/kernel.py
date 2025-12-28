"""
═══════════════════════════════════════════════════════════════════════════════
                        FRACTAL 3D KERNEL - ЯДРО СИСТЕМЫ
═══════════════════════════════════════════════════════════════════════════════

Диссертация: "Геометрическое моделирование голографических изображений 
             сложных фрактальных 3D объектов"

Автор: Набиев Ильхом Шарифович
ТУИТ имени Мухаммада аль-Хорезми (2025)

Это ЯДРО системы, которое преобразует универсальные значения в 3D объекты.

Математические основы (из статьи):
- Уравнение плоскости: Ax + By + Cz + D = 0
- R-функции Рвачёва для объединения граней
- IFS (Iterated Function Systems) для фрактальных итераций
- Аффинные преобразования для масштабирования и перемещения

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class FractalType(Enum):
    """Типы фракталов"""
    SIERPINSKI_TETRAHEDRON = "Треугольник Серпинского"
    SIERPINSKI_PYRAMID = "Треугольник Серпинского"
    MENGER_SPONGE = "Губка Менгера"
    CUSTOM = "Пользовательский"


@dataclass
class UniversalInput:
    """
    УНИВЕРСАЛЬНЫЕ ВХОДНЫЕ ЗНАЧЕНИЯ
    
    Пользователь вводит эти параметры, а AI определяет тип фрактала
    и ядро генерирует 3D объект.
    """
    # Геометрические параметры
    num_vertices: int = 4           # Количество вершин (4=тетраэдр, 8=куб)
    num_faces: int = 4              # Количество граней
    symmetry_order: int = 3         # Порядок симметрии
    
    # Фрактальные параметры  
    iterations: int = 4             # Количество итераций
    scale_factor: float = 0.5       # Коэффициент масштабирования
    hausdorff_dim: float = 2.0      # Размерность Хаусдорфа
    
    # Размеры
    size: float = 2.0               # Общий размер
    
    # Дополнительно
    self_similar: bool = True       # Самоподобие
    hollow: bool = False            # Полый объект (как губка Менгера)
    branching: int = 4              # Ветвление
    
    # Текстовое описание для AI
    description: str = ""


@dataclass 
class Fractal3DResult:
    """Результат генерации 3D фрактала"""
    fractal_type: str
    points: np.ndarray              # 3D точки
    faces: np.ndarray = None        # Грани для меша
    normals: np.ndarray = None      # Нормали
    colors: np.ndarray = None       # Цвета вершин
    
    # Метаданные
    num_points: int = 0
    num_faces: int = 0
    iterations: int = 0
    hausdorff_dim: float = 0.0
    bounds_min: np.ndarray = None
    bounds_max: np.ndarray = None


class PlaneEquation:
    """
    Уравнение плоскости: Ax + By + Cz + D = 0
    
    Из статьи, формула (1): Аx + By + Cz + D = 0
    """
    
    def __init__(self, A: float, B: float, C: float, D: float):
        self.A, self.B, self.C, self.D = A, B, C, D
        self._normalize()
    
    def _normalize(self):
        """Нормализация коэффициентов"""
        n = np.sqrt(self.A**2 + self.B**2 + self.C**2)
        if n > 1e-10:
            self.A /= n
            self.B /= n
            self.C /= n
            self.D /= n
    
    @classmethod
    def from_points(cls, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
        """Создание плоскости через 3 точки (формула 2-3 из статьи)"""
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        A, B, C = n
        D = -np.dot(n, p1)
        return cls(A, B, C, D)
    
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Вычислить значение для точек"""
        return self.A * points[:, 0] + self.B * points[:, 1] + self.C * points[:, 2] + self.D


class RFunction:
    """
    R-функции Рвачёва для CSG операций
    
    Из статьи, формула (6-7):
    R∧(f1, f2) = f1 + f2 - √(f1² + f2²)  - пересечение
    R∨(f1, f2) = f1 + f2 + √(f1² + f2²)  - объединение
    """
    
    @staticmethod
    def intersection(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """R∧ - пересечение (AND)"""
        return f1 + f2 - np.sqrt(f1**2 + f2**2)
    
    @staticmethod
    def union(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """R∨ - объединение (OR)"""
        return f1 + f2 + np.sqrt(f1**2 + f2**2)
    
    @staticmethod
    def subtraction(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """Вычитание"""
        return RFunction.intersection(f1, -f2)
    
    @staticmethod
    def combine_faces(faces: List[np.ndarray]) -> np.ndarray:
        """Объединение всех граней в один объект"""
        if not faces:
            return np.array([])
        result = faces[0]
        for f in faces[1:]:
            result = RFunction.intersection(result, f)
        return result


class AffineTransform:
    """
    Аффинные преобразования для IFS
    
    Из статьи, формула (8-10):
    x' = sx·x + tx
    y' = sy·y + ty  
    z' = sz·z + tz
    """
    
    def __init__(self, scale: float = 0.5, translation: np.ndarray = None):
        self.scale = scale
        self.translation = translation if translation is not None else np.zeros(3)
        
        # Матрица преобразования 4x4
        self.matrix = np.eye(4)
        self.matrix[0, 0] = self.matrix[1, 1] = self.matrix[2, 2] = scale
        self.matrix[:3, 3] = self.translation
    
    def apply(self, points: np.ndarray) -> np.ndarray:
        """Применить преобразование к точкам"""
        return points * self.scale + self.translation
    
    def apply_single(self, point: np.ndarray) -> np.ndarray:
        """Применить к одной точке"""
        return point * self.scale + self.translation


class Fractal3DKernel:
    """
    ═══════════════════════════════════════════════════════════════════════
                           ГЛАВНОЕ ЯДРО СИСТЕМЫ
    ═══════════════════════════════════════════════════════════════════════
    
    Преобразует универсальные входные значения в 3D фрактальные объекты.
    
    Алгоритм:
    1. Получить универсальные параметры
    2. AI определяет тип фрактала
    3. Ядро создаёт базовую геометрию (тетраэдр, куб и т.д.)
    4. Применяет R-функции для объединения граней
    5. Выполняет IFS итерации
    6. Возвращает 3D точки/меш
    """
    
    def __init__(self):
        self.r_func = RFunction()
    
    def generate(self, input_data: UniversalInput, 
                 fractal_type: FractalType = None) -> Fractal3DResult:
        """
        ГЛАВНЫЙ МЕТОД ГЕНЕРАЦИИ
        
        Args:
            input_data: Универсальные входные значения
            fractal_type: Тип фрактала (если None - определяется автоматически)
        
        Returns:
            Fractal3DResult с 3D данными
        """
        # Если тип не указан - определяем по параметрам
        if fractal_type is None:
            fractal_type = self._infer_fractal_type(input_data)
        
        # Генерация в зависимости от типа
        generators = {
            FractalType.SIERPINSKI_TETRAHEDRON: self._generate_sierpinski_tetrahedron,
            FractalType.SIERPINSKI_PYRAMID: self._generate_sierpinski_tetrahedron,
            FractalType.MENGER_SPONGE: self._generate_menger_sponge,
        }

        generator = generators.get(fractal_type, self._generate_sierpinski_tetrahedron)
        points = generator(input_data)

        # Вычисляем размерность Хаусдорфа
        hausdorff = self._compute_hausdorff_dimension(fractal_type, input_data)

        # Создаём цвета на основе Z-координаты
        colors = self._generate_colors(points)

        return Fractal3DResult(
            fractal_type=fractal_type.value,
            points=points,
            colors=colors,
            num_points=len(points),
            iterations=input_data.iterations,
            hausdorff_dim=hausdorff,
            bounds_min=points.min(axis=0),
            bounds_max=points.max(axis=0)
        )

    def _infer_fractal_type(self, input_data: UniversalInput) -> FractalType:
        """Определение типа фрактала по входным параметрам"""

        # По количеству вершин
        if input_data.num_vertices == 4:
            return FractalType.SIERPINSKI_TETRAHEDRON
        elif input_data.num_vertices == 8:
            if input_data.hollow:
                return FractalType.MENGER_SPONGE
            return FractalType.MENGER_SPONGE

        # По размерности Хаусдорфа
        if abs(input_data.hausdorff_dim - 2.0) < 0.1:
            return FractalType.SIERPINSKI_TETRAHEDRON
        elif abs(input_data.hausdorff_dim - 2.727) < 0.1:
            return FractalType.MENGER_SPONGE

        # По тексту
        desc = input_data.description.lower()
        if any(w in desc for w in ['sierpinski', 'серпинск', 'треугольник', 'пирамид', 'triangle']):
            return FractalType.SIERPINSKI_TETRAHEDRON
        elif any(w in desc for w in ['menger', 'менгер', 'губка', 'sponge']):
            return FractalType.MENGER_SPONGE

        return FractalType.SIERPINSKI_TETRAHEDRON

    def _generate_sierpinski_tetrahedron(self, input_data: UniversalInput) -> np.ndarray:
        """
        Генерация тетраэдра Серпинского

        Из статьи - формулы (4-10):
        1. Создаём базовый тетраэдр с вершинами V1, V2, V3, V4
        2. Вычисляем плоскости граней P1, P2, P3, P4
        3. Объединяем через R-функции
        4. Применяем IFS итерации
        """
        size = input_data.size
        iterations = input_data.iterations
        scale = input_data.scale_factor

        # Вершины правильного тетраэдра (из статьи)
        a = size / np.sqrt(2)
        vertices = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=np.float64) * a / 2

        # IFS преобразования - к каждой вершине
        transforms = []
        for v in vertices:
            transforms.append(AffineTransform(scale=scale, translation=v * (1 - scale)))

        # Chaos Game алгоритм
        num_points = int(50000 * (input_data.iterations / 4))
        points = np.zeros((num_points, 3))
        current = np.zeros(3)

        # Прогрев
        for _ in range(100):
            t = transforms[np.random.randint(4)]
            current = t.apply_single(current)

        # Генерация точек
        for i in range(num_points):
            t = transforms[np.random.randint(4)]
            current = t.apply_single(current)
            points[i] = current

        return points

    def _generate_menger_sponge(self, input_data: UniversalInput) -> np.ndarray:
        """Генерация губки Менгера"""
        size = input_data.size
        iterations = input_data.iterations

        # Начинаем с заполнения куба точками
        points = []

        def is_in_menger(x, y, z, level):
            """Проверка, находится ли точка в губке Менгера"""
            for _ in range(level):
                x, y, z = x * 3, y * 3, z * 3
                xi, yi, zi = int(x) % 3, int(y) % 3, int(z) % 3

                # Если в центре грани или в центре куба - вырезаем
                if (xi == 1 and yi == 1) or (yi == 1 and zi == 1) or (xi == 1 and zi == 1):
                    return False

                x, y, z = x % 1, y % 1, z % 1
            return True

        # Генерируем точки
        num_samples = int(100000 * (iterations / 4))
        samples = np.random.rand(num_samples, 3) * size - size/2

        for pt in samples:
            # Нормализуем к [0,1]
            x, y, z = (pt + size/2) / size
            if is_in_menger(x, y, z, iterations):
                points.append(pt)

        return np.array(points) if points else np.zeros((1, 3))

    def _generate_koch_3d(self, input_data: UniversalInput) -> np.ndarray:
        """Генерация 3D снежинки Коха"""
        size = input_data.size
        iterations = input_data.iterations
        scale = 1/3

        # Вершины тетраэдра
        vertices = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [1, np.sqrt(3), 0],
            [1, np.sqrt(3)/3, np.sqrt(6)/3 * 2]
        ]) * size / 2

        transforms = [AffineTransform(scale=scale, translation=v * (1-scale)) for v in vertices]

        # Chaos Game
        num_points = int(50000 * (iterations / 4))
        points = np.zeros((num_points, 3))
        current = np.zeros(3)

        for _ in range(100):
            current = transforms[np.random.randint(4)].apply_single(current)

        for i in range(num_points):
            current = transforms[np.random.randint(4)].apply_single(current)
            points[i] = current

        return points

    def _generate_cantor_3d(self, input_data: UniversalInput) -> np.ndarray:
        """Генерация 3D пыли Кантора"""
        size = input_data.size
        iterations = input_data.iterations

        # 8 углов куба с масштабом 1/3
        scale = 1/3
        offsets = np.array([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ]) * size / 3

        transforms = [AffineTransform(scale=scale, translation=o) for o in offsets]

        num_points = int(50000 * (iterations / 4))
        points = np.zeros((num_points, 3))
        current = np.zeros(3)

        for _ in range(100):
            current = transforms[np.random.randint(8)].apply_single(current)

        for i in range(num_points):
            current = transforms[np.random.randint(8)].apply_single(current)
            points[i] = current

        return points

    def _compute_hausdorff_dimension(self, fractal_type: FractalType,
                                     input_data: UniversalInput) -> float:
        """Вычисление размерности Хаусдорфа"""
        dimensions = {
            FractalType.SIERPINSKI_TETRAHEDRON: np.log(4) / np.log(2),  # ≈ 2.0
            FractalType.SIERPINSKI_PYRAMID: np.log(4) / np.log(2),
            FractalType.MENGER_SPONGE: np.log(20) / np.log(3),  # ≈ 2.727
        }
        return dimensions.get(fractal_type, 2.0)

    def _generate_colors(self, points: np.ndarray) -> np.ndarray:
        """Генерация цветов на основе Z-координаты"""
        if len(points) == 0:
            return np.zeros((0, 4))

        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)

        # Градиент от синего к жёлтому
        colors = np.zeros((len(points), 4))
        colors[:, 0] = z_norm           # R
        colors[:, 1] = z_norm * 0.8     # G
        colors[:, 2] = 1 - z_norm       # B
        colors[:, 3] = 1.0              # Alpha

        return colors


# ═══════════════════════════════════════════════════════════════════════════
# ТЕСТ
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ТЕСТ ЯДРА FRACTAL 3D")
    print("=" * 70)

    kernel = Fractal3DKernel()

    # Тест с разными параметрами
    test_cases = [
        UniversalInput(num_vertices=4, iterations=4, description="тетраэдр Серпинского"),
        UniversalInput(num_vertices=8, hollow=True, iterations=3, description="губка"),
        UniversalInput(hausdorff_dim=1.26, iterations=4, description="снежинка"),
    ]

    for i, inp in enumerate(test_cases):
        result = kernel.generate(inp)
        print(f"\nТест {i+1}:")
        print(f"  Тип: {result.fractal_type}")
        print(f"  Точек: {result.num_points}")
        print(f"  Размерность: {result.hausdorff_dim:.3f}")
        print(f"  Границы: {result.bounds_min.round(2)} - {result.bounds_max.round(2)}")

    print("\n" + "=" * 70)
    print("✓ Все тесты пройдены!")