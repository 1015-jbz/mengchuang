# 🚗 智能座舱多模态交互终端 — "小航"

> 基于 LoongArch 端侧 AI | 语音对话 | 表情识别 | 大模型推理 | 安全驾驶守护

---

## 给别人用（最简方式）

### 零、前提
- Windows 10/11，Python 3.10+
- 摄像头（表情识别用）+ 麦克风（语音输入可选）

### 一、安装环境（首次）
双击 **`setup.bat`**，自动完成：创建虚拟环境 → 安装依赖 → 检查模型文件。

### 二、启动
双击 **`start.bat`**，浏览器打开 **http://localhost:7860**。

### 三、其他人从局域网访问
启动后，同一 WiFi 下的设备访问 `http://你的电脑IP:7860`（启动时终端会打印 IP 地址）。

---

## 功能一览

| 模块 | 说明 |
|------|------|
| 😊 表情识别 | ONNX EfficientNet-B2，7 种表情 + 悲伤增强后处理 |
| 🎤 语音输入 | faster-whisper tiny 模型（D 盘缓存），支持中文语音转文字 |
| 🤖 智能对话 | DeepSeek 大模型优先（多轮记忆 + 情绪感知），离线模板兜底 |
| 🎙️ 语音合成 | edge-tts 8 种中文语音可选（动漫/方言/男女声），pyttsx3 离线兜底 |
| 🎚️ 声线 DIY | 音高偏移 + 语速调节滑块，一键懒羊羊风格预设 |
| 🛡️ 安全监控 | PERCLOS 疲劳检测 + 分心检测 + 四级告警（CLI 模式） |
| 🎛️ 座舱控制 | NL 控制空调/车窗/座椅/灯光/导航/音乐（CLI 模式） |

## 项目结构

```
smart_cockpit/
├── app_demo.py          # Web Demo 主入口（Gradio + Flask MJPEG）
├── main.py              # CLI 模式入口
├── setup.bat            # 一键安装脚本
├── start.bat            # 一键启动脚本
├── .env                 # DeepSeek API Key（不提交 Git）
├── requirements.txt     # Python 依赖
├── configs/settings.py  # 全局配置
├── src/                 # 核心引擎（orchestrator/asr/nlu/llm/safety/emotion/control）
├── models/              # ONNX 表情模型
├── tests/               # 集成测试
└── docs/                # 设计文档
```

## 手动启动

```bash
cd D:\Claude_Workspace\smart_cockpit
.venv\Scripts\activate    # 如果有虚拟环境
python app_demo.py
# → http://localhost:7860
```

## 环境变量

在 `.env` 文件中配置：
```
DEEPSEEK_API_KEY=sk-xxxxxxxx
```
不配也能跑（自动回退本地模板对话）。

## 远程仓库

https://github.com/1015-jbz/mengchuang
