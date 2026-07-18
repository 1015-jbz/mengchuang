@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   智能座舱 - 一键环境安装
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo 下载: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [√] Python 已安装

:: 创建虚拟环境
if not exist ".venv" (
    echo [*] 创建虚拟环境到 .venv...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

:: 安装依赖
echo [*] 安装 Python 依赖...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 检查模型文件
echo.
if not exist "models\enet_b2_7.onnx" (
    echo [!] 表情识别模型未下载 (models\enet_b2_7.onnx)
    echo    请从交接文档中获取下载方式
) else (
    echo [√] 表情识别模型就绪
)

:: 检查 Whisper 模型
if exist "D:\huggingface_cache\models--Systran--faster-whisper-tiny" (
    echo [√] Whisper 语音模型就绪
) else (
    echo [!] Whisper 模型未下载，语音输入将降级为文本
    echo    运行: python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')"
)

:: 检查 .env
if exist ".env" (
    echo [√] DeepSeek API Key 已配置
) else (
    echo [!] 未配置 .env 文件，大模型对话将使用本地模板
    echo    创建 .env 文件并写入: DEEPSEEK_API_KEY=你的key
)

echo.
echo ========================================
echo   安装完成！双击 start.bat 启动
echo ========================================
pause
