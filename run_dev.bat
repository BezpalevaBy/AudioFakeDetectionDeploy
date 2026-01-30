@echo off
setlocal

echo ===============================
echo Deepfake Audio Detector (DEV)
echo ===============================

REM Создание venv
if not exist venv (
    python -m venv venv
)

REM Активация venv
call venv\Scripts\activate.bat

REM Обновление pip
python -m pip install --upgrade pip

REM Установка зависимостей
pip install -r requirements-dev.txt

REM Запуск сервера
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
