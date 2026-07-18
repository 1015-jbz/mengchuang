@echo off
cd /d "%~dp0"

echo ========================================
echo   Smart Cockpit - Xiao Hang
echo ========================================
echo.
echo   Open: http://localhost:7860
echo   Stop: Ctrl+C
echo ========================================
echo.

call .venv\Scripts\activate.bat 2>nul
python app_demo.py
pause
