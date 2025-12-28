@echo off
chcp 65001 >nul
title Создание Fractal3D.exe

echo ╔══════════════════════════════════════════════════════════════╗
echo ║          СОЗДАНИЕ FRACTAL3D.EXE                              ║
echo ║          Автор: Набиев И.Ш. • ТУИТ 2025                      ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo [1/4] Проверка Python...
python --version
if errorlevel 1 (
    echo ОШИБКА: Python не установлен!
    echo Скачайте с https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo [2/4] Установка зависимостей...
pip install numpy matplotlib PyQt6 pyinstaller --quiet

echo.
echo [3/4] Создание EXE файла (это займет 2-5 минут)...
pyinstaller --onefile --windowed --name Fractal3D --icon=icon.ico --add-data "core;core" --add-data "ai;ai" --add-data "gui;gui" --add-data "exports;exports" run.py

if errorlevel 1 (
    echo.
    echo Попробуем без иконки...
    pyinstaller --onefile --windowed --name Fractal3D --add-data "core;core" --add-data "ai;ai" --add-data "gui;gui" --add-data "exports;exports" run.py
)

echo.
echo [4/4] Готово!
echo.

if exist "dist\Fractal3D.exe" (
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║  ✅ УСПЕХ! Файл создан: dist\Fractal3D.exe                   ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo Откройте папку dist и запустите Fractal3D.exe
    explorer dist
) else (
    echo ❌ Ошибка создания EXE
)

echo.
pause
