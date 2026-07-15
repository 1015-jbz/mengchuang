# 🚗 基于 LoongArch 端侧AI的智能座舱多模态交互终端

> **参赛作品** — 智能汽车座舱 · 安全驾驶守护 · 情绪化陪伴

---

## 项目简介

本项目设计并实现了一套**完全运行在端侧**（龙芯 LoongArch 架构）的智能座舱多模态交互系统。不同于依赖云端的方案，本系统将语音识别、意图理解、大模型推理、疲劳检测和情绪识别全部部署在车规级龙芯终端上，实现**零网络依赖、隐私全保护、毫秒级响应**。

### 三大核心场景

| 场景 | 功能 | 价值 |
|------|------|------|
| 🧭 **智能语音秘书** | 语音控制导航/空调/车窗/音乐，解放双手 | 驾驶安全 |
| 🛡️ **安全驾驶守护** | 疲劳检测、分心预警、危险行为识别 | 生命守护 |
| 💬 **情绪化陪伴** | 表情识别、情绪感知、共情对话、主动关怀 | 情感慰藉 |

---

## 快速开始 (Windows)

### 1. 环境准备

```bash
# Python 3.10+ 推荐
python --version

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动可视化 Demo（推荐）

```bash
# 启动 Gradio 界面（表情识别 + 语音对话）
python app_demo.py
# 浏览器打开 http://localhost:7860
```

**功能体验:**
- 📷 摄像头实时**表情识别**（开心/悲伤/愤怒/惊讶/恐惧）
- 🎤 麦克风**语音对话**（ASR识别 → 意图理解 → LLM回复 → TTS播报）
- 😊 根据表情自动切换**共情回复风格**
- 🛡️ 疲劳/分心状态实时**安全监控**

### 3. 文本模式运行（无需摄像头/麦克风）

```bash
python main.py
# 输入文本对话，按 q 退出
```

---

## 项目结构

```
smart_cockpit/
├── app_demo.py                  # 🎯 Gradio 可视化Demo（一站式体验）
├── main.py                      # 命令行模式主入口
├── requirements.txt             # Python 依赖
├── README.md
├── configs/
│   └── settings.py              # 全局配置（模型/阈值/参数）
├── src/
│   ├── orchestrator.py          # 🧠 核心编排器（事件路由+上下文融合）
│   ├── asr/
│   │   └── speech_engine.py     # 🎤 语音引擎（ASR+TTS+VAD+唤醒词）
│   ├── nlu/
│   │   └── intent_engine.py     # 🎯 意图理解（30+意图分类+槽位填充）
│   ├── llm/
│   │   └── llm_engine.py        # 🤖 端侧大模型（llama.cpp+RAG知识库）
│   ├── safety/
│   │   └── safety_monitor.py    # 🛡️ 安全监控（PERCLOS疲劳+分心+危险预警）
│   ├── emotion/
│   │   └── emotion_engine.py    # 😊 情感计算（表情+语音+文本三模态融合）
│   ├── control/
│   │   └── cockpit_control.py   # 🎛️ 座舱控制（NL车控+场景模式+安全限制）
│   ├── multimodal/
│   │   └── fusion.py            # 🔗 多模态融合（语音+视觉+车辆信号）
│   └── utils/
│       └── event_bus.py         # 📡 异步事件总线
├── data/
│   ├── knowledge/               # RAG 知识库文件
│   └── samples/                 # 音频样本
├── models/                      # 模型文件目录
├── tests/                       # 单元测试
└── docs/
    └── 01_总体设计方案.md         # 📄 详细设计方案文档
```

---

## 软件功能清单

### M1 语音交互模块
- [x] 语音唤醒（自定义唤醒词）
- [x] 流式语音识别 (ASR)
- [x] 语音活动检测 (VAD)
- [x] 语音合成 (TTS)
- [x] 声纹识别（主驾/副驾区分）

### M2 意图理解模块
- [x] 6大领域分类（导航/车控/娱乐/安全/陪伴/信息）
- [x] 30+ 细粒度意图识别
- [x] 槽位填充（目的地/温度/歌曲名等）
- [x] 多轮对话管理
- [x] 模糊意图消歧

### M3 端侧大模型模块
- [x] GGUF 量化模型推理 (llama.cpp)
- [x] RAG 知识库检索增强
- [x] 流式 Token 输出
- [x] 多轮对话记忆（滑动窗口+摘要压缩）
- [x] 安全护栏（敏感话题过滤）

### M4 驾驶安全监控
- [x] 疲劳检测（PERCLOS + 哈欠频率 + 头部姿态）
- [x] 分心检测（视线偏离 + 持续时长）
- [x] 四级风险告警（normal/warning/high/critical）
- [x] 驾驶时长统计

### M5 情感计算引擎
- [x] 面部表情识别（7种基础表情）
- [x] 文本情感分析
- [x] 情绪趋势追踪
- [x] 主动情感关怀
- [x] 共情对话生成

### M6 座舱控制
- [x] 自然语言车控（空调/车窗/座椅/灯光）
- [x] 行驶中安全限制
- [x] 场景模式引擎

---

## 技术架构

```
┌──────────────────────────────────────────┐
│        Gradio / 车载HMI 交互层            │
├──────────────────────────────────────────┤
│          CockpitOrchestrator              │
│     (上下文融合 + 意图路由 + 安全决策)      │
├──────────┬──────────┬────────────────────┤
│  ASR/TTS │  NLU     │  LLM (llama.cpp)   │
│  Whisper │  BERT    │  Qwen2.5-7B-GGUF   │
├──────────┼──────────┼────────────────────┤
│  Safety  │  Emotion │  Cockpit Control   │
│  Monitor │  Engine  │  (CAN Middleware)   │
├──────────┴──────────┴────────────────────┤
│          Event Bus (ZeroMQ)               │
├──────────────────────────────────────────┤
│   硬件: 龙芯 LoongArch + NPU + DSP        │
└──────────────────────────────────────────┘
```

---

## LoongArch 移植指南

### 关键依赖的 LoongArch 适配

| 库 | Windows | LoongArch 方案 |
|----|---------|---------------|
| PyTorch | ✅ pip | LoongArch 源码编译或龙芯社区预编译包 |
| llama.cpp | ✅ pip | 源码编译 + OpenBLAS (LoongArch 优化) |
| ONNX Runtime | ✅ pip | 龙芯官方适配版本 |
| OpenCV | ✅ pip | LoongArch 源码编译 |
| faster-whisper | ✅ pip | 替换为 whisper.cpp 编译版本 |
| edge-tts | ✅ pip | 替换为 espeak/pyttsx3 离线TTS |

### 编译 llama.cpp for LoongArch

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release -j$(nproc)
```

---

## 开发计划

| 阶段 | 时间 | 内容 |
|------|------|------|
| **Phase 1** | 第1-4周 | Windows Python 原型开发 |
| **Phase 2** | 第5-6周 | 功能联调、Gradio Demo |
| **Phase 3** | 第7-8周 | LoongArch 虚拟机移植测试 |

---

## 依赖安装问题 FAQ

### DeepFace 安装失败
```bash
# DeepFace 依赖较多，可先跳过用 opencv 方案
pip install deepface
# 如果失败，app_demo.py 会自动降级到 OpenCV 基础检测
```

### faster-whisper 安装失败
```bash
# 替代方案
pip install openai-whisper
```

### CUDA 相关错误
```bash
# 本项目默认使用 CPU 推理，忽略 CUDA 相关警告
# 或在 settings.py 中设置 device="cpu"
```

---

## 参赛信息

- **题目**: 基于LoongArch端侧AI的智能座舱多模态交互终端设计与实现
- **硬件平台**: 龙芯 LoongArch (3A6000 / 2K2000)
- **关键词**: 端侧AI · 智能座舱 · 多模态交互 · 安全驾驶 · 情感计算
