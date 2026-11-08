@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Menyiapkan Python virtual environment 3.11...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Python 3.11 tidak ditemukan. Silakan pasang Python 3.11 terlebih dahulu.
        pause
        exit /b 1
    )
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Instalasi gagal. Periksa koneksi internet dan versi Python.
    pause
    exit /b 1
)
echo.
echo Buka http://127.0.0.1:5000 di browser Anda.
python app.py
pause
