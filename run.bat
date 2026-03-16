@echo off
cd /d "%~dp0"
python --version >nul 2>&1 || (echo Python 3.8+ required & pause & exit /b)
if not exist "venv\" python -m venv venv
call venv\Scripts\activate.bat
pip install --quiet flask numpy pandas scikit-learn matplotlib scipy
python -c "import torch" 2>nul || pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
echo Open: http://localhost:5000
python app.py
pause
