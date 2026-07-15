@echo off
REM ============================================
REM 智能座舱多模态交互终端 — 快速启动脚本
REM ============================================
echo.
echo ==========================================
echo   智能座舱多模态交互终端
echo   Smart Cockpit Setup
echo ==========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 未安装或不在 PATH 中
    echo 请安装 Python 3.10+
    pause
    exit /b 1
)

echo [1/3] 创建虚拟环境...
if not exist "venv" (
    python -m venv venv
    echo   虚拟环境已创建
) else (
    echo   虚拟环境已存在
)

echo [2/3] 安装依赖...
call venv\Scripts\activate
pip install -r requirements.txt --quiet

echo [3/3] 安装完成！
echo.
echo ==========================================
echo   启动方式:
echo.
echo   1. Gradio 可视化 Demo (推荐):
echo      python app_demo.py
echo      → 浏览器打开 http://localhost:7860
echo.
echo   2. 命令行模式:
echo      python main.py
echo.
echo   3. 仅测试表情识别:
echo      python -c "from src.emotion.emotion_engine import EmotionEngine; print('OK')"
echo ==========================================
echo.
pause
