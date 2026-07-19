# 🚗 智能座舱多模态交互终端 — "小航"

> 表情识别 | 驾驶安全监控 | 智能语音对话 | 声线 DIY
>
> **本地推理**：表情识别（ONNX）、安全监控（PERCLOS/哈欠）、ASR（faster-whisper）
> **云端能力**：对话（DeepSeek API，可降级本地模板）、TTS（edge-tts，可降级 pyttsx3）
> **目标平台**：LoongArch 端侧 AI（当前 Demo 在 Windows x86 开发，待硬件到位后迁移）

---

## 🚀 快速开始（Windows 用户）

### 方式一：一键安装（推荐）

1. **克隆仓库**
   ```bash
   git clone https://github.com/1015-jbz/mengchuang.git
   cd mengchuang
   ```

2. **双击 `setup.bat`**
   - 自动创建 Python 虚拟环境
   - 自动安装所有依赖（使用清华源加速）
   - 自动从 GitHub Release 下载 ONNX 表情识别模型（~30MB）
   - 自动复制 `.env.example` 为 `.env` 模板

3. **配置 DeepSeek API Key**（可选，不配也能跑）
   - 编辑 `.env` 文件，填入你的 key
   - 申请地址：https://platform.deepseek.com/api_keys

4. **双击 `start.bat` 启动**
   - 浏览器自动打开 http://localhost:7860
   - 同一 WiFi 下其他设备访问 `http://你的电脑IP:7860`

### 方式二：手动命令行

```bash
# 1. 克隆仓库
git clone https://github.com/1015-jbz/mengchuang.git
cd mengchuang

# 2. 创建虚拟环境（需要 Python 3.10+）
python -m venv .venv
.venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 下载 ONNX 表情识别模型
python scripts\setup_models.py

# 5. 配置 API Key
copy .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-你的key

# 6. 启动
python app_demo.py
```

---

## 📋 前置条件

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Windows 10/11 | Linux/Mac 需手动改路径分隔符 |
| Python | 3.10+ | [下载地址](https://www.python.org/downloads/) |
| 摄像头 | 必需 | 表情识别用 |
| 麦克风 | 可选 | 语音输入用，没有也能用文字 |
| 网络 | 首次需要 | 下载依赖和模型；之后可离线运行（对话降级为模板） |

---

## 📦 自动下载的文件

首次运行 `setup.bat` 时会自动下载以下文件，无需手动操作：

| 文件 | 大小 | 用途 | 来源 |
|------|------|------|------|
| `models/enet_b2_7.onnx` | ~30MB | ONNX 表情识别模型 | GitHub Release |
| `~/.cache/huggingface/...` | ~75MB | faster-whisper tiny 语音模型 | HuggingFace（首次用语音输入时自动下载） |

> 如果 GitHub Release 下载失败，可手动从 [Releases 页面](https://github.com/1015-jbz/mengchuang/releases) 下载 `enet_b2_7.onnx`，放到 `models/` 目录下。

---

## 🎯 功能一览

| 模块 | 说明 | 推理位置 |
|------|------|---------|
| 😊 表情识别 | ONNX EfficientNet-B2，7 种表情 + 悲伤增强后处理 | 本地 |
| 🛡️ 安全监控 | PERCLOS 疲劳检测 + 分心检测 + 四级告警（CLI 模式） | 本地 |
| 🎤 语音输入 | faster-whisper tiny，支持中文语音转文字 | 本地 |
| 🤖 智能对话 | DeepSeek 大模型优先（多轮记忆 + 情绪感知），离线模板兜底 | 云端 + 本地兜底 |
| 🎙️ 语音合成 | edge-tts 8 种中文语音可选（动漫/方言/男女声），pyttsx3 离线兜底 | 云端 + 本地兜底 |
| 🎚️ 声线 DIY | 音高偏移 + 语速调节滑块，一键懒羊羊风格预设 | — |
| 🎛️ 座舱控制 | NL 控制空调/车窗/座椅/灯光/导航/音乐（CLI 模式） | — |

---

## 📁 项目结构

```
mengchuang/
├── app_demo.py              # Web Demo 主入口（Gradio + Flask MJPEG）
├── main.py                  # CLI 模式入口
├── setup.bat                # 一键安装脚本
├── start.bat                # 一键启动脚本
├── .env.example             # 环境变量模板（复制为 .env 使用）
├── .env                     # 你的 API Key 配置（不提交 Git）
├── requirements.txt         # Python 依赖
├── configs/settings.py      # 全局配置
├── scripts/
│   └── setup_models.py      # ONNX 模型自动下载脚本
├── src/                     # 核心引擎（orchestrator/asr/nlu/llm/safety/emotion/control）
├── models/                  # ONNX 表情模型（自动下载，不入 Git）
├── tests/                   # 集成测试
└── docs/                    # 设计文档
```

---

## ⚙️ 环境变量

在 `.env` 文件中配置（首次安装会从 `.env.example` 复制模板）：

```bash
# 必填：DeepSeek API Key（不配也能跑，对话降级为本地模板）
DEEPSEEK_API_KEY=sk-你的key

# 可选：显式指定 faster-whisper 模型路径
# 不设置则按以下顺序查找：本地 models/whisper-tiny → HuggingFace 自动下载
# WHISPER_MODEL_PATH=/path/to/whisper-tiny
```

申请 DeepSeek API Key：https://platform.deepseek.com/api_keys

---

## 🔧 故障排查

### Q: `setup.bat` 下载 ONNX 模型失败？
- 浏览器打开 https://github.com/1015-jbz/mengchuang/releases
- 手动下载 `enet_b2_7.onnx`
- 放到 `models/` 目录下

### Q: 语音输入无法识别？
- 首次使用语音输入时，faster-whisper 会自动下载模型（~75MB），需要联网
- 如果 HuggingFace 下载失败，可手动下载后设置环境变量 `WHISPER_MODEL_PATH` 指向模型目录

### Q: 没配置 DeepSeek API Key 能用吗？
- 能用，但智能对话会降级为本地模板回复（功能有限）
- 建议申请 key 后填入 `.env` 体验完整功能

### Q: 启动后浏览器看不到视频画面？
- 检查摄像头是否被其他程序占用
- 浏览器控制台查看是否有 `ERR_UNKNOWN_URL_SCHEME` 之外的报错
- 视频流直连地址：http://localhost:7861/video_feed

### Q: 其他人如何访问？
- 启动后终端会打印局域网 IP（如 `http://192.168.x.x:7860`）
- 同一 WiFi 下设备直接访问该地址即可
- 公网访问需要 ngrok 隧道（启动时会自动建立）

---

## 🌐 远程仓库

https://github.com/1015-jbz/mengchuang

如遇问题，欢迎提 Issue。
