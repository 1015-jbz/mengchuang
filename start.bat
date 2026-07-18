@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   智能座舱多模态交互终端
echo   启动中...
echo ========================================
echo.
echo   浏览器打开: http://localhost:7860
echo   按 Ctrl+C 停止
echo ========================================
echo.

call .venv\Scripts\activate.bat 2>nul
python app_demo.py
pause
