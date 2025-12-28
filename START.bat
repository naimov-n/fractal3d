@echo off
chcp 65001 >nul
title Fractal3D - Запуск

echo Запуск Fractal3D...
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден!
    echo.
    echo Установите Python:
    echo 1. Откройте https://www.python.org/downloads/
    echo 2. Скачайте и установите Python
    echo 3. ВАЖНО: Поставьте галочку "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

REM Проверка PyQt6
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo Установка зависимостей...
    pip install numpy matplotlib PyQt6 --quiet
)

REM Запуск
python run.py

if errorlevel 1 (
    echo.
    echo Ошибка запуска. Нажмите любую клавишу...
    pause >nul
)
