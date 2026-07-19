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

:: 升级 pip
echo [*] 升级 pip...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 安装依赖
echo.
echo [*] 安装 Python 依赖（使用清华源加速）...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败，请检查网络或 requirements.txt
    pause
    exit /b 1
)
echo [√] Python 依赖已安装

:: 下载 ONNX 表情识别模型（自动从 GitHub Release 下载）
echo.
echo [*] 下载 ONNX 表情识别模型（~30MB）...
if not exist "models\enet_b2_7.onnx" (
    python scripts\setup_models.py
    if not exist "models\enet_b2_7.onnx" (
        echo [警告] 自动下载失败，请参考 scripts\setup_models.py 中的手动方案
    ) else (
        echo [√] ONNX 模型下载完成
    )
) else (
    echo [√] ONNX 表情识别模型已就绪
)

:: 检查 .env
echo.
if exist ".env" (
    echo [√] DeepSeek API Key 已配置
) else (
    echo [!] 未配置 .env 文件
    echo    正在从 .env.example 复制模板...
    if exist ".env.example" copy .env.example .env >nul
    echo.
    echo    请编辑 .env 文件，填入你的 DeepSeek API Key
    echo    申请地址: https://platform.deepseek.com/api_keys
    echo.
    echo    未配置 key 也能运行，但智能对话会降级为本地模板回复
)

echo.
echo ========================================
echo   安装完成！
echo.
echo   下一步:
echo     1. 编辑 .env 填入 DeepSeek API Key（可选）
echo     2. 双击 start.bat 启动程序
echo ========================================
echo.
echo Whisper 语音模型会在首次使用语音输入时自动下载（~75MB）
echo.
pause
