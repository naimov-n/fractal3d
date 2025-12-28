#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════

    ███████╗██████╗  █████╗  ██████╗████████╗ █████╗ ██╗      ██████╗ ██████╗ 
    ██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██║      ╚════██╗██╔══██╗
    █████╗  ██████╔╝███████║██║        ██║   ███████║██║       █████╔╝██║  ██║
    ██╔══╝  ██╔══██╗██╔══██║██║        ██║   ██╔══██║██║       ╚═══██╗██║  ██║
    ██║     ██║  ██║██║  ██║╚██████╗   ██║   ██║  ██║███████╗ ██████╔╝██████╔╝
    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ 

═══════════════════════════════════════════════════════════════════════════════

FRACTAL3D - ПРОФЕССИОНАЛЬНЫЙ ГЕНЕРАТОР 3D ФРАКТАЛОВ

Диссертация: "Геометрическое моделирование голографических изображений 
              сложных фрактальных 3D объектов"

Автор: Набиев Ильхом Шарифович
Научный руководитель: Нуралиев Ф.М.
ТУИТ имени Мухаммада аль-Хорезми (2025)

═══════════════════════════════════════════════════════════════════════════════

Использование:
    python run.py           # Запуск GUI приложения
    python run.py --cli     # Командная строка
    python run.py --help    # Справка

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import argparse

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Проверка зависимостей"""
    required = ['numpy', 'matplotlib']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Установка зависимостей: {missing}")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing + ['-q'])


def run_gui():
    """Запуск GUI приложения"""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("Установка PyQt6...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'PyQt6', '-q'])
    
    from gui.main_window import main
    main()


def run_cli(args):
    """Командная строка"""
    from core.kernel import Fractal3DKernel, UniversalInput
    from exports.exporter import ExportManager
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║               FRACTAL3D - КОМАНДНАЯ СТРОКА                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    kernel = Fractal3DKernel()
    
    inp = UniversalInput(
        description=args.type,
        iterations=args.iterations,
        size=args.size
    )
    
    print(f"Генерация: {args.type}")
    print(f"Итераций: {args.iterations}")
    
    result = kernel.generate(inp)
    
    print(f"✓ Создано {result.num_points:,} точек")
    print(f"✓ Тип: {result.fractal_type}")
    print(f"✓ Размерность: {result.hausdorff_dim:.3f}")
    
    if args.output:
        path = ExportManager.export(result.points, args.output, args.format, colors=result.colors)
        print(f"✓ Сохранено: {path}")
    
    print("\nГотово!")


def main():
    parser = argparse.ArgumentParser(
        description='Fractal3D - Генератор 3D Фракталов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python run.py                                    # GUI
  python run.py --cli -t sierpinski -o fractal.obj # CLI
  python run.py --cli -t "губка менгера" -i 3      # На русском
        """
    )
    
    parser.add_argument('--cli', action='store_true', help='Режим командной строки')
    parser.add_argument('-t', '--type', default='sierpinski', help='Тип фрактала')
    parser.add_argument('-i', '--iterations', type=int, default=4, help='Итерации')
    parser.add_argument('-s', '--size', type=float, default=2.0, help='Размер')
    parser.add_argument('-o', '--output', help='Выходной файл')
    parser.add_argument('-f', '--format', default='obj', 
                       choices=['obj', 'stl', 'ply', 'gltf', 'xyz', 'csv'],
                       help='Формат экспорта')
    
    args = parser.parse_args()
    
    check_dependencies()
    
    if args.cli:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
