"""
智能座舱多模态交互终端 — 全局配置
"""
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import os

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
CONFIGS_DIR = ROOT_DIR / "configs"


class ASRConfig(BaseModel):
    """语音识别配置"""
    model: str = "tiny"               # tiny / base / small / medium
    language: str = "zh"              # 识别语言
    device: str = "cpu"               # cpu / cuda
    compute_type: str = "int8"        # float32 / int8
    sample_rate: int = 16000
    vad_aggressiveness: int = 2       # 0-3，VAD 激进程度
    wake_word: str = "你好小航"        # 唤醒词
    wake_model_path: Optional[str] = None


class NLUConfig(BaseModel):
    """意图理解配置"""
    intent_model: str = "bert-base-chinese"
    device: str = "cpu"
    confidence_threshold: float = 0.65  # 意图置信度阈值
    max_context_turns: int = 10          # 多轮对话窗口
    slot_filling_model: str = "bert-base-chinese"


class LLMConfig(BaseModel):
    """大模型推理配置"""
    model_path: str = ""                # GGUF 模型文件路径
    n_ctx: int = 4096                   # 上下文长度
    n_threads: int = 4                  # CPU 线程数
    n_batch: int = 512                  # 批处理大小
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    # RAG 配置
    rag_enabled: bool = True
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    vector_db_path: str = str(DATA_DIR / "vector_db")


class SafetyConfig(BaseModel):
    """安全监控配置"""
    camera_id: int = 0
    # 疲劳检测阈值
    perclos_threshold: float = 0.3      # 眼睑闭合比例
    yawn_frequency_threshold: float = 3.0  # 打哈欠频率（次/分钟）
    # 分心检测
    gaze_deviation_threshold: float = 30.0  # 视线偏离角度
    distraction_duration_threshold: float = 2.0  # 持续分心秒数
    # 驾驶时长
    max_continuous_driving_min: int = 120  # 最长连续驾驶分钟


class EmotionConfig(BaseModel):
    """情感计算配置"""
    face_emotion_model: str = "hustvl/yolos-tiny"
    speech_emotion_model: str = "facebook/wav2vec2-base"
    text_emotion_model: str = "bert-base-chinese"
    # 情绪追踪
    emotion_history_size: int = 50       # 保留最近N次情绪记录
    proactive_check_interval_min: int = 10  # 主动情绪检查间隔


class SystemConfig(BaseModel):
    """系统总配置"""
    asr: ASRConfig = ASRConfig()
    nlu: NLUConfig = NLUConfig()
    llm: LLMConfig = LLMConfig()
    safety: SafetyConfig = SafetyConfig()
    emotion: EmotionConfig = EmotionConfig()
    # 系统
    log_level: str = "INFO"
    event_bus_address: str = "ipc:///tmp/smart_cockpit_bus"
    # TTS
    tts_engine: str = "edge"             # edge / pyttsx3 / offline
    tts_voice: str = "zh-CN-XiaoxiaoNeural"


# 全局配置实例
config = SystemConfig()
