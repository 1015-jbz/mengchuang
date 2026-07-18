"""
语音引擎 — ASR语音识别 + TTS语音合成 + 唤醒词检测
Windows: faster-whisper + edge-tts
LoongArch: whisper.cpp + 本地TTS
"""
import asyncio
from typing import Optional, Callable
from pathlib import Path

import numpy as np

from loguru import logger
from src.utils.event_bus import EventBus, Event
from configs.settings import ASRConfig, config as system_config


class SpeechEngine:
    """语音引擎 — 端侧语音交互核心"""

    def __init__(self, event_bus: EventBus, config: ASRConfig):
        self.event_bus = event_bus
        self.config = config
        self.is_listening = False
        self.asr_model = None
        self.wake_model = None
        self.vad = None
        self._edge_warned = False

    async def initialize(self):
        """初始化 ASR/TTS/VAD 模型"""
        logger.info("加载语音识别模型...")

        # ----- VAD (语音活动检测) -----
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)
            logger.info("  VAD 初始化成功 (WebRTC)")
        except ImportError:
            logger.warning("  webrtcvad 未安装，使用能量VAD降级方案")
            self.vad = EnergyVAD()

        # ----- ASR (语音识别) -----
        # 注意: 模型未缓存时 WhisperModel 会尝试联网下载，受限网络下会失败/卡死，
        # 因此用 local_files_only + 宽异常捕获，失败则降级为文本输入而不是让启动崩溃。
        # 优先查 D 盘手动下载路径（企业网 SSL 拦截，模型已手工下载到 D 盘）
        try:
            from faster_whisper import WhisperModel
            _d_path = "D:/huggingface_cache/models--Systran--faster-whisper-tiny/snapshots/tiny"
            from pathlib import Path as _Path
            model_arg = _d_path if _Path(_d_path).is_dir() else self.config.model
            self.asr_model = WhisperModel(
                model_arg,
                device=self.config.device,
                compute_type=self.config.compute_type,
                local_files_only=True,
            )
            logger.info(f"  ASR 模型加载成功: faster-whisper ({self.config.model})")
        except ImportError:
            logger.warning("  faster-whisper 未安装，尝试 openai-whisper")
            try:
                import whisper
                self.asr_model = whisper.load_model(self.config.model)
                logger.info(f"  ASR 模型加载成功: whisper ({self.config.model})")
            except Exception as e:
                logger.warning(f"  openai-whisper 不可用 ({e})，ASR 降级为文本输入")
                self.asr_model = None
        except Exception as e:
            logger.warning(f"  ASR 模型未就绪 ({e})")
            logger.info(f"  请先下载 whisper {self.config.model} 模型；当前降级为文本输入")
            self.asr_model = None

        # ----- 唤醒词检测 -----
        logger.info(f"唤醒词: '{self.config.wake_word}'")
        # TODO: 集成 Porcupine / Snowboy / 自训练唤醒词模型
        # 当前使用模拟唤醒（按键触发）
        self.wake_detected = False

        # ----- 麦克风 -----
        self.audio_stream = None
        logger.info("语音引擎初始化完成")

    async def listen_loop(self):
        """
        持续监听循环

        流程:
        1. 麦克风采集音频
        2. VAD 检测语音活动
        3. 语音片段送入 ASR 识别
        4. 识别结果发布到事件总线
        """
        if self.asr_model is None:
            logger.info("ASR 模型不可用，进入文本输入模式")
            await self._text_fallback_loop()
            return

        try:
            import pyaudio
            self.audio = pyaudio.PyAudio()
            self.audio_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=480,  # 30ms@16kHz — webrtcvad 仅接受 10/20/30ms 帧长
            )
            logger.info("🎤 麦克风已就绪，开始监听...")
        except Exception as e:
            logger.error(f"麦克风初始化失败: {e}")
            logger.info("将以模拟模式运行（文本输入代替语音）")
            # 降级到文本输入模式
            await self._text_fallback_loop()
            return

        self.is_listening = True
        audio_buffer = []
        silence_frames = 0
        SPEECH_START_THRESHOLD = 10   # 连续检测到语音帧开始录音
        SILENCE_STOP_THRESHOLD = 30   # 连续静音帧停止录音

        while self.is_listening:
            try:
                # 读取音频帧
                data = self.audio_stream.read(480, exception_on_overflow=False)
                is_speech = self.vad.is_speech(data, self.config.sample_rate)

                if is_speech:
                    audio_buffer.append(data)
                    silence_frames = 0
                elif audio_buffer:
                    # 语音后的静音
                    silence_frames += 1
                    if silence_frames > SILENCE_STOP_THRESHOLD:
                        # 语音段结束，送入 ASR
                        audio_data = b"".join(audio_buffer)
                        text = await self._transcribe(audio_data)
                        if text:
                            await self.event_bus.publish(Event(
                                type="speech.recognized",
                                data={"text": text, "speaker": "driver", "source": "microphone"}
                            ))
                        audio_buffer = []
                        silence_frames = 0

                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"音频采集异常: {e}")
                await asyncio.sleep(0.1)

    async def _transcribe(self, audio_data: bytes) -> Optional[str]:
        """将音频数据转录为文本"""
        if self.asr_model is None:
            return None
        try:
            # 将原始 PCM 转为 numpy 数组
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if isinstance(self.asr_model, object) and hasattr(self.asr_model, 'transcribe'):
                # faster-whisper
                segments, _ = self.asr_model.transcribe(
                    audio_np,
                    language=self.config.language,
                    beam_size=5,
                )
                text = " ".join(seg.text for seg in segments).strip()
            else:
                # openai-whisper (同步调用)
                text = self.asr_model.transcribe(
                    audio_np,
                    language=self.config.language,
                )["text"].strip()

            if text:
                logger.debug(f"ASR: {text}")
            return text

        except Exception as e:
            logger.error(f"ASR 识别失败: {e}")
            return None

    async def speak(self, text: str, interrupt: bool = False):
        """
        TTS 语音合成并播放（线程池执行，不阻塞事件循环）

        引擎由 configs.settings 的 tts_engine 决定:
        - pyttsx3: 离线合成+播放（Windows SAPI，LoongArch 上可换 espeak），CLI 默认
        - edge:    edge-tts 输出为 MP3，CLI 无本地解码器暂无法播放 → 自动回退 pyttsx3
                   （Web Demo 的 edge-tts 语音由浏览器播放，见 app_demo.py）
        注: interrupt 打断当前播报尚未实现（TODO）
        """
        if not text:
            return

        logger.info(f"🔊 TTS: {text}")

        if system_config.tts_engine == "edge" and not self._edge_warned:
            self._edge_warned = True
            logger.warning("edge-tts 输出 MP3，CLI 下无解码器无法播放，已改用 pyttsx3 离线播报")

        def _run():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                logger.warning(f"TTS 播报失败({e})，降级为文本输出")
                print(f"\n🤖 [小航]: {text}\n")

        await asyncio.get_event_loop().run_in_executor(None, _run)

    async def _text_fallback_loop(self):
        """降级模式 — 文本输入代替语音"""
        logger.info("📝 文本输入模式（输入 'q' 退出）")
        self.is_listening = True   # 修复: 此前未置 True，下面的循环体一次都不会执行
        while self.is_listening:
            try:
                # 在线程池中运行阻塞的 input
                text = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\n💬 你说: ")
                )
                if text.lower() in ('q', 'quit', 'exit'):
                    self.is_listening = False
                    break
                if text.strip():
                    await self.event_bus.publish(Event(
                        type="speech.recognized",
                        data={"text": text.strip(), "speaker": "driver", "source": "text"}
                    ))
            except EOFError:
                await asyncio.sleep(1)

    async def shutdown(self):
        """关闭语音引擎"""
        self.is_listening = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        logger.info("语音引擎已关闭")


class EnergyVAD:
    """基于能量的简单 VAD（webrtcvad 的降级方案）"""
    def __init__(self, threshold: float = 500.0):
        self.threshold = threshold

    def is_speech(self, data: bytes, sample_rate: int) -> bool:
        audio = np.frombuffer(data, dtype=np.int16)
        energy = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        return energy > self.threshold
