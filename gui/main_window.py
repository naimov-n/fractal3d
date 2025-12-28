#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    FRACTAL3D PRO - DESKTOP ПРИЛОЖЕНИЕ
═══════════════════════════════════════════════════════════════════════════════

Автор: Набиев Ильхом Шарифович
ТУИТ имени Мухаммада аль-Хорезми (2025)

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QProgressBar, QSlider, 
    QCheckBox, QFrame, QGridLayout, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPalette, QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.kernel import Fractal3DKernel, UniversalInput, Fractal3DResult
from ai.recognizer import FractalRecognizer
from exports.exporter import ExportManager


STYLE = """
QMainWindow, QWidget { background-color: #1a1a2e; color: #e8e8e8; font-family: 'Segoe UI'; }
QGroupBox { border: 2px solid #0f3460; border-radius: 10px; margin-top: 15px; padding-top: 10px; font-weight: bold; }
QGroupBox::title { color: #00d9ff; left: 15px; padding: 0 10px; }
QPushButton { background: #0f3460; border: none; border-radius: 8px; padding: 12px 25px; font-weight: bold; color: white; }
QPushButton:hover { background: #e94560; }
QPushButton#genBtn { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e94560, stop:1 #0f3460); font-size: 16px; padding: 15px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit { background: #0f3460; border: 2px solid #1a508b; border-radius: 8px; padding: 8px; color: white; }
QSlider::groove:horizontal { height: 8px; background: #0f3460; border-radius: 4px; }
QSlider::handle:horizontal { background: #e94560; width: 18px; margin: -5px 0; border-radius: 9px; }
QProgressBar { border: 2px solid #1a508b; border-radius: 8px; text-align: center; }
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e94560, stop:1 #00d9ff); }
"""


class Viewer3D(FigureCanvas):
    """3D Просмотрщик"""
    
    def __init__(self):
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor='#1a1a2e')
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
        self.ax.set_xlabel('X', color='#00d9ff')
        self.ax.set_ylabel('Y', color='#00d9ff')
        self.ax.set_zlabel('Z', color='#00d9ff')
        self.points = None
        self.rotation = [30, 45]
        self.timer = QTimer()
        self.timer.timeout.connect(self._rotate)
    
    def display(self, points, colors=None, title=""):
        self.points = points
        self.ax.clear()
        self.ax.set_facecolor('#1a1a2e')
        
        if len(points) > 0:
            pts = points[::max(1, len(points)//20000)]
            c = colors[::max(1, len(colors)//20000)][:, :3] if colors is not None else pts[:, 2]
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=0.5, alpha=0.7, cmap='viridis')
            self.ax.set_title(title, color='#00d9ff', fontsize=12)
        
        self.ax.view_init(elev=self.rotation[0], azim=self.rotation[1])
        self.draw()
    
    def set_rotation(self, elev, azim):
        self.rotation = [elev, azim]
        if self.points is not None:
            self.ax.view_init(elev=elev, azim=azim)
            self.draw()
    
    def toggle_rotate(self, on):
        if on:
            self.timer.start(50)
        else:
            self.timer.stop()
    
    def _rotate(self):
        self.rotation[1] = (self.rotation[1] + 2) % 360
        if self.points is not None:
            self.ax.view_init(elev=self.rotation[0], azim=self.rotation[1])
            self.draw()


class GeneratorThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, kernel, input_data):
        super().__init__()
        self.kernel = kernel
        self.input_data = input_data
    
    def run(self):
        try:
            self.progress.emit(20, "AI распознавание...")
            self.progress.emit(50, "Генерация 3D...")
            result = self.kernel.generate(self.input_data)
            self.progress.emit(100, "Готово!")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.kernel = Fractal3DKernel()
        self.recognizer = FractalRecognizer()
        self.result = None
        
        self.setWindowTitle("Fractal3D - Генератор 3D Фракталов | Набиев И.Ш.")
        self.setMinimumSize(1300, 850)
        self.setStyleSheet(STYLE)
        self._setup_ui()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setSpacing(20)
        main.setContentsMargins(20, 20, 20, 20)
        
        # ЛЕВАЯ ПАНЕЛЬ
        left = QWidget()
        left.setFixedWidth(380)
        left_layout = QVBoxLayout(left)
        
        # Заголовок
        title = QLabel("🔮 FRACTAL3D")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #00d9ff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)
        
        author = QLabel("Диссертация: Набиев И.Ш. • ТУИТ 2025")
        author.setStyleSheet("color: #888; font-size: 11px;")
        author.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(author)
        
        # Описание
        desc_grp = QGroupBox("📝 Описание (AI распознаёт)")
        desc_lay = QVBoxLayout(desc_grp)
        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Введите: Треугольник Серпинского, Губка Менгера...")
        self.desc_edit.setMaximumHeight(80)
        desc_lay.addWidget(self.desc_edit)
        left_layout.addWidget(desc_grp)
        
        # Параметры
        param_grp = QGroupBox("⚙️ Универсальные параметры")
        param_grid = QGridLayout(param_grp)
        
        param_grid.addWidget(QLabel("Вершины:"), 0, 0)
        self.vertices = QSpinBox()
        self.vertices.setRange(3, 20)
        self.vertices.setValue(4)
        param_grid.addWidget(self.vertices, 0, 1)
        
        param_grid.addWidget(QLabel("Итерации:"), 1, 0)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 8)
        self.iterations.setValue(4)
        param_grid.addWidget(self.iterations, 1, 1)
        
        param_grid.addWidget(QLabel("Размерность:"), 2, 0)
        self.dimension = QDoubleSpinBox()
        self.dimension.setRange(1.0, 3.0)
        self.dimension.setValue(2.0)
        self.dimension.setSingleStep(0.1)
        param_grid.addWidget(self.dimension, 2, 1)
        
        param_grid.addWidget(QLabel("Размер:"), 3, 0)
        self.size = QDoubleSpinBox()
        self.size.setRange(0.5, 10.0)
        self.size.setValue(2.0)
        param_grid.addWidget(self.size, 3, 1)
        
        self.hollow = QCheckBox("Полый объект")
        param_grid.addWidget(self.hollow, 4, 0, 1, 2)
        
        left_layout.addWidget(param_grp)
        
        # Пресеты
        preset_grp = QGroupBox("🚀 Быстрый выбор")
        preset_lay = QVBoxLayout(preset_grp)
        self.preset = QComboBox()
        self.preset.addItems(["-- Пресет --", "Треугольник Серпинского", "Губка Менгера"])
        self.preset.currentIndexChanged.connect(self._on_preset)
        preset_lay.addWidget(self.preset)
        left_layout.addWidget(preset_grp)
        
        # Кнопка генерации
        self.gen_btn = QPushButton("🔮 СГЕНЕРИРОВАТЬ 3D")
        self.gen_btn.setObjectName("genBtn")
        self.gen_btn.clicked.connect(self._generate)
        left_layout.addWidget(self.gen_btn)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)
        
        self.info = QLabel("")
        self.info.setStyleSheet("color: #888;")
        self.info.setWordWrap(True)
        left_layout.addWidget(self.info)
        
        left_layout.addStretch()
        main.addWidget(left)
        
        # ПРАВАЯ ПАНЕЛЬ
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        # 3D Вид
        view_grp = QGroupBox("🎮 3D Просмотр")
        view_lay = QVBoxLayout(view_grp)
        self.viewer = Viewer3D()
        self.viewer.setMinimumHeight(450)
        view_lay.addWidget(self.viewer)
        
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Обзор:"))
        self.elev = QSlider(Qt.Orientation.Horizontal)
        self.elev.setRange(0, 90)
        self.elev.setValue(30)
        self.elev.valueChanged.connect(lambda: self.viewer.set_rotation(self.elev.value(), self.azim.value()))
        ctrl.addWidget(self.elev)
        
        self.azim = QSlider(Qt.Orientation.Horizontal)
        self.azim.setRange(0, 360)
        self.azim.setValue(45)
        self.azim.valueChanged.connect(lambda: self.viewer.set_rotation(self.elev.value(), self.azim.value()))
        ctrl.addWidget(self.azim)
        
        self.auto_rot = QCheckBox("Авто")
        self.auto_rot.toggled.connect(self.viewer.toggle_rotate)
        ctrl.addWidget(self.auto_rot)
        
        view_lay.addLayout(ctrl)
        right_layout.addWidget(view_grp)
        
        # Экспорт
        exp_grp = QGroupBox("💾 Экспорт 3D")
        exp_lay = QHBoxLayout(exp_grp)
        
        self.exp_fmt = QComboBox()
        for k, v in ExportManager.get_format_info().items():
            self.exp_fmt.addItem(f"{v['name']} ({v['extension']})", k)
        exp_lay.addWidget(self.exp_fmt)
        
        self.exp_btn = QPushButton("📥 Скачать")
        self.exp_btn.setEnabled(False)
        self.exp_btn.clicked.connect(self._export)
        exp_lay.addWidget(self.exp_btn)
        
        self.exp_all = QPushButton("📦 Все форматы")
        self.exp_all.setEnabled(False)
        self.exp_all.clicked.connect(self._export_all)
        exp_lay.addWidget(self.exp_all)
        
        right_layout.addWidget(exp_grp)
        
        # Статистика
        stat_grp = QGroupBox("📊 Статистика")
        stat_lay = QVBoxLayout(stat_grp)
        self.stats = QLabel("Сгенерируйте фрактал")
        self.stats.setWordWrap(True)
        stat_lay.addWidget(self.stats)
        right_layout.addWidget(stat_grp)
        
        main.addWidget(right, 1)
        
        self.statusBar().showMessage("Готов")
    
    def _on_preset(self, idx):
        presets = {
            1: ("Треугольник Серпинского", 4, 4, 2.0, False),
            2: ("Губка Менгера", 8, 3, 2.727, True),
        }
        if idx in presets:
            n, v, i, d, h = presets[idx]
            self.desc_edit.setText(n)
            self.vertices.setValue(v)
            self.iterations.setValue(i)
            self.dimension.setValue(d)
            self.hollow.setChecked(h)
    
    def _generate(self):
        self.gen_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        inp = UniversalInput(
            num_vertices=self.vertices.value(),
            iterations=self.iterations.value(),
            hausdorff_dim=self.dimension.value(),
            size=self.size.value(),
            hollow=self.hollow.isChecked(),
            description=self.desc_edit.toPlainText()
        )
        
        self.thread = GeneratorThread(self.kernel, inp)
        self.thread.progress.connect(lambda v, m: (self.progress.setValue(v), self.statusBar().showMessage(m)))
        self.thread.finished.connect(self._on_done)
        self.thread.error.connect(self._on_error)
        self.thread.start()
    
    def _on_done(self, result):
        self.result = result
        self.viewer.display(result.points, result.colors, result.fractal_type)
        
        self.stats.setText(f"""
        <b>Тип:</b> {result.fractal_type}<br>
        <b>Точек:</b> {result.num_points:,}<br>
        <b>Итераций:</b> {result.iterations}<br>
        <b>Размерность:</b> {result.hausdorff_dim:.3f}
        """)
        
        self.exp_btn.setEnabled(True)
        self.exp_all.setEnabled(True)
        self.gen_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage("✅ Готово!")
    
    def _on_error(self, msg):
        self.gen_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Ошибка", msg)
    
    def _export(self):
        if not self.result:
            return
        fmt = self.exp_fmt.currentData()
        ext = ExportManager.get_format_info()[fmt]['extension']
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить", f"fractal{ext}")
        if path:
            try:
                ExportManager.export(self.result.points, path, fmt, colors=self.result.colors)
                QMessageBox.information(self, "Успех", f"Сохранено: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
    
    def _export_all(self):
        if not self.result:
            return
        folder = QFileDialog.getExistingDirectory(self, "Папка для экспорта")
        if folder:
            try:
                base = os.path.join(folder, "fractal")
                results = ExportManager.export_all(self.result.points, base, colors=self.result.colors)
                ok = [f for f, p in results.items() if "Ошибка" not in str(p)]
                QMessageBox.information(self, "Экспорт", f"Экспортировано: {len(ok)} файлов")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(232, 232, 232))
    palette.setColor(QPalette.ColorRole.Base, QColor(15, 52, 96))
    palette.setColor(QPalette.ColorRole.Text, QColor(232, 232, 232))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(233, 69, 96))
    app.setPalette(palette)
    
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
