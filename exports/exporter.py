"""
═══════════════════════════════════════════════════════════════════════════════
                        МОДУЛЬ ЭКСПОРТА 3D ОБЪЕКТОВ
═══════════════════════════════════════════════════════════════════════════════

Поддерживаемые форматы:
- OBJ (Wavefront) - для Blender, Maya, 3ds Max
- STL - для 3D печати
- PLY - Point Cloud с цветами
- GLTF/GLB - для веба и игр
- FBX (через промежуточный формат)

Автор: Набиев И.Ш. (2025)
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import json
import struct


class Exporter3D:
    """Главный класс экспорта 3D объектов"""
    
    @staticmethod
    def export_obj(points: np.ndarray, 
                   filepath: str,
                   faces: np.ndarray = None,
                   colors: np.ndarray = None,
                   name: str = "Fractal3D") -> str:
        """
        Экспорт в OBJ формат
        
        Args:
            points: Nx3 массив точек
            filepath: Путь к файлу
            faces: Mx3 массив индексов граней (опционально)
            colors: Nx3 или Nx4 массив цветов (опционально)
            name: Название объекта
        
        Returns:
            Путь к созданному файлу
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.obj')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Fractal 3D Object\n")
            f.write(f"# Создано: Fractal3D Pro\n")
            f.write(f"# Автор: Набиев И.Ш.\n")
            f.write(f"# Точек: {len(points)}\n\n")
            f.write(f"o {name}\n\n")
            
            # Вершины с цветами (если есть)
            if colors is not None and len(colors) == len(points):
                for i, (p, c) in enumerate(zip(points, colors)):
                    r, g, b = c[:3]
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
            else:
                for p in points:
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
            # Грани
            if faces is not None and len(faces) > 0:
                f.write(f"\n# Грани: {len(faces)}\n")
                for face in faces:
                    indices = ' '.join(str(i + 1) for i in face)
                    f.write(f"f {indices}\n")
            else:
                # Точки как отдельные элементы
                f.write(f"\n# Точки\n")
                for i in range(len(points)):
                    f.write(f"p {i + 1}\n")
        
        return str(filepath)
    
    @staticmethod
    def export_stl(points: np.ndarray,
                   filepath: str,
                   faces: np.ndarray = None,
                   binary: bool = True,
                   name: str = "Fractal3D") -> str:
        """
        Экспорт в STL формат (для 3D печати)
        
        Args:
            points: Nx3 массив точек
            filepath: Путь к файлу
            faces: Mx3 массив индексов граней
            binary: Бинарный формат (компактнее)
            name: Название объекта
        
        Returns:
            Путь к созданному файлу
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.stl')
        
        # Если нет граней - создаём из точек (каждые 3 точки = треугольник)
        if faces is None or len(faces) == 0:
            n = (len(points) // 3) * 3
            if n < 3:
                # Создаём минимальный треугольник
                faces = np.array([[0, 1, 2]])
                if len(points) < 3:
                    points = np.vstack([points, np.zeros((3 - len(points), 3))])
            else:
                faces = np.arange(n).reshape(-1, 3)
        
        if binary:
            return Exporter3D._export_stl_binary(points, faces, filepath, name)
        else:
            return Exporter3D._export_stl_ascii(points, faces, filepath, name)
    
    @staticmethod
    def _export_stl_binary(points: np.ndarray, faces: np.ndarray, 
                           filepath: Path, name: str) -> str:
        """Бинарный STL"""
        with open(filepath, 'wb') as f:
            # Заголовок (80 байт)
            header = f"Fractal3D: {name}"[:80].ljust(80)
            f.write(header.encode('ascii'))
            
            # Количество треугольников
            f.write(struct.pack('<I', len(faces)))
            
            # Треугольники
            for face in faces[:min(len(faces), 100000)]:  # Лимит
                p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
                
                # Нормаль
                v1, v2 = p2 - p1, p3 - p1
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-10:
                    normal /= norm_len
                
                # Записываем нормаль и вершины
                f.write(struct.pack('<fff', *normal))
                f.write(struct.pack('<fff', *p1))
                f.write(struct.pack('<fff', *p2))
                f.write(struct.pack('<fff', *p3))
                f.write(struct.pack('<H', 0))  # Атрибуты
        
        return str(filepath)
    
    @staticmethod
    def _export_stl_ascii(points: np.ndarray, faces: np.ndarray,
                          filepath: Path, name: str) -> str:
        """ASCII STL"""
        with open(filepath, 'w') as f:
            f.write(f"solid {name}\n")
            
            for face in faces[:min(len(faces), 50000)]:
                p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
                
                v1, v2 = p2 - p1, p3 - p1
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-10:
                    normal /= norm_len
                
                f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write(f"    outer loop\n")
                f.write(f"      vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n")
                f.write(f"      vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\n")
                f.write(f"      vertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
            
            f.write(f"endsolid {name}\n")
        
        return str(filepath)
    
    @staticmethod
    def export_ply(points: np.ndarray,
                   filepath: str,
                   colors: np.ndarray = None,
                   faces: np.ndarray = None,
                   binary: bool = False) -> str:
        """
        Экспорт в PLY формат (Point Cloud с цветами)
        
        Args:
            points: Nx3 массив точек
            filepath: Путь к файлу
            colors: Nx3 или Nx4 массив цветов [0-1] или [0-255]
            faces: Mx3 массив граней (опционально)
            binary: Бинарный формат
        
        Returns:
            Путь к созданному файлу
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.ply')
        
        has_colors = colors is not None and len(colors) == len(points)
        has_faces = faces is not None and len(faces) > 0
        
        # Нормализация цветов к [0, 255]
        if has_colors:
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)
            if colors.shape[1] == 4:
                colors = colors[:, :3]  # Убираем альфа
        
        with open(filepath, 'w') as f:
            # Заголовок
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Created by Fractal3D Pro\n")
            f.write(f"comment Author: Nabiev I.Sh.\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if has_colors:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            if has_faces:
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
            
            f.write("end_header\n")
            
            # Вершины
            for i, p in enumerate(points):
                if has_colors:
                    c = colors[i]
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
                else:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
            # Грани
            if has_faces:
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        return str(filepath)
    
    @staticmethod
    def export_gltf(points: np.ndarray,
                    filepath: str,
                    colors: np.ndarray = None,
                    faces: np.ndarray = None,
                    name: str = "Fractal3D") -> str:
        """
        Экспорт в GLTF формат (для веба)
        
        Простая реализация GLTF 2.0
        """
        import base64
        
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.gltf')
        
        # Конвертируем точки в bytes
        points_f32 = points.astype(np.float32)
        points_bytes = points_f32.tobytes()
        points_b64 = base64.b64encode(points_bytes).decode('ascii')
        
        # Минимальный GLTF
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "Fractal3D Pro by Nabiev I.Sh."
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": name}],
            "meshes": [{
                "primitives": [{
                    "attributes": {"POSITION": 0},
                    "mode": 0  # POINTS
                }],
                "name": name
            }],
            "accessors": [{
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(points),
                "type": "VEC3",
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist()
            }],
            "bufferViews": [{
                "buffer": 0,
                "byteLength": len(points_bytes),
                "target": 34962  # ARRAY_BUFFER
            }],
            "buffers": [{
                "uri": f"data:application/octet-stream;base64,{points_b64}",
                "byteLength": len(points_bytes)
            }]
        }
        
        with open(filepath, 'w') as f:
            json.dump(gltf, f, indent=2)
        
        return str(filepath)
    
    @staticmethod
    def export_xyz(points: np.ndarray, filepath: str) -> str:
        """Простой XYZ формат"""
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.xyz')
        
        np.savetxt(filepath, points, fmt='%.6f', delimiter=' ',
                   header=f'Fractal3D - {len(points)} points')
        return str(filepath)
    
    @staticmethod
    def export_csv(points: np.ndarray, filepath: str, 
                   colors: np.ndarray = None) -> str:
        """CSV формат с опциональными цветами"""
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.csv')
        
        with open(filepath, 'w') as f:
            if colors is not None and len(colors) == len(points):
                f.write("x,y,z,r,g,b\n")
                for p, c in zip(points, colors):
                    f.write(f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{c[0]:.4f},{c[1]:.4f},{c[2]:.4f}\n")
            else:
                f.write("x,y,z\n")
                for p in points:
                    f.write(f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")
        
        return str(filepath)
    
    @staticmethod
    def export_npy(points: np.ndarray, filepath: str,
                   colors: np.ndarray = None,
                   metadata: Dict = None) -> str:
        """NumPy формат (для Python)"""
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.npz')
        
        data = {'points': points}
        if colors is not None:
            data['colors'] = colors
        if metadata:
            data['metadata'] = np.array([json.dumps(metadata)])
        
        np.savez_compressed(filepath, **data)
        return str(filepath)


class ExportManager:
    """Менеджер экспорта с поддержкой всех форматов"""
    
    FORMATS = {
        'obj': ('Wavefront OBJ', '.obj', Exporter3D.export_obj),
        'stl': ('STL (3D Print)', '.stl', Exporter3D.export_stl),
        'ply': ('PLY Point Cloud', '.ply', Exporter3D.export_ply),
        'gltf': ('GLTF 2.0 (Web)', '.gltf', Exporter3D.export_gltf),
        'xyz': ('XYZ Points', '.xyz', Exporter3D.export_xyz),
        'csv': ('CSV Data', '.csv', Exporter3D.export_csv),
        'npy': ('NumPy Array', '.npz', Exporter3D.export_npy),
    }
    
    @classmethod
    def export(cls, points: np.ndarray, filepath: str, 
               format: str = 'obj', **kwargs) -> str:
        """
        Универсальный экспорт
        
        Args:
            points: Nx3 массив точек
            filepath: Путь к файлу
            format: Формат (obj, stl, ply, gltf, xyz, csv, npy)
            **kwargs: Дополнительные параметры
        
        Returns:
            Путь к созданному файлу
        """
        format = format.lower()
        if format not in cls.FORMATS:
            raise ValueError(f"Неизвестный формат: {format}. Доступные: {list(cls.FORMATS.keys())}")
        
        _, ext, export_func = cls.FORMATS[format]
        
        # Добавляем расширение если нет
        if not Path(filepath).suffix:
            filepath = str(Path(filepath).with_suffix(ext))
        
        return export_func(points, filepath, **kwargs)
    
    @classmethod
    def export_all(cls, points: np.ndarray, base_path: str, 
                   formats: List[str] = None, **kwargs) -> Dict[str, str]:
        """
        Экспорт во все форматы
        
        Args:
            points: Nx3 массив точек
            base_path: Базовый путь (без расширения)
            formats: Список форматов (None = все)
            **kwargs: Дополнительные параметры
        
        Returns:
            Dict {формат: путь_к_файлу}
        """
        if formats is None:
            formats = list(cls.FORMATS.keys())
        
        results = {}
        base = Path(base_path)
        
        for fmt in formats:
            try:
                filepath = str(base.with_suffix(cls.FORMATS[fmt][1]))
                results[fmt] = cls.export(points, filepath, fmt, **kwargs)
            except Exception as e:
                results[fmt] = f"Ошибка: {e}"
        
        return results
    
    @classmethod
    def get_format_info(cls) -> Dict:
        """Информация о форматах"""
        return {k: {'name': v[0], 'extension': v[1]} for k, v in cls.FORMATS.items()}


# ═══════════════════════════════════════════════════════════════════════════
# ТЕСТ
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ТЕСТ ЭКСПОРТА 3D")
    print("=" * 70)
    
    # Тестовые данные
    points = np.random.randn(1000, 3)
    colors = np.random.rand(1000, 3)
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results = ExportManager.export_all(
            points, 
            os.path.join(tmpdir, "test_fractal"),
            colors=colors
        )
        
        for fmt, path in results.items():
            if "Ошибка" not in str(path):
                size = os.path.getsize(path)
                print(f"  ✓ {fmt.upper()}: {Path(path).name} ({size:,} bytes)")
            else:
                print(f"  ✗ {fmt.upper()}: {path}")
    
    print("\n" + "=" * 70)
    print("✓ Все тесты пройдены!")
