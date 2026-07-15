"""
情感计算引擎 — 表情识别 + 语音情绪 + 文本情感 + 共情对话
多模态融合：面部表情(ViT/CNN) + 语音语调(Wav2Vec2) + 文本语义(BERT)
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque

import numpy as np

from loguru import logger
from src.utils.event_bus import EventBus, Event
from configs.settings import EmotionConfig


@dataclass
class EmotionState:
    """情绪状态快照"""
    # 面部表情
    face_emotion: str = "neutral"
    face_confidence: float = 0.0
    # 语音情绪
    speech_emotion: str = "neutral"
    speech_confidence: float = 0.0
    # 文本情感
    text_sentiment: str = "neutral"
    text_confidence: float = 0.0
    # 融合结果
    dominant_emotion: str = "neutral"
    fusion_confidence: float = 0.0
    # 长期追踪
    emotion_trend: str = "stable"    # stable / improving / declining
    needs_care: bool = False
    # 时间戳
    timestamp: float = 0.0


# 情绪标签（中英文映射）
EMOTION_LABELS_ZH = {
    "happy": "开心", "sad": "悲伤", "angry": "愤怒",
    "surprised": "惊讶", "fearful": "恐惧", "disgusted": "厌恶",
    "neutral": "平静", "calm": "平静", "anxious": "焦虑",
    "excited": "兴奋", "tired": "疲倦", "bored": "无聊",
}


class EmotionEngine:
    """
    情感计算引擎

    三大模态:
    1. 面部表情识别 — 基于 CNN/ViT (OpenCV + 轻量模型)
    2. 语音情绪识别 — 韵律特征 + Wav2Vec2
    3. 文本情感分析 — BERT 情感分类

    LoongArch: 全部转为 ONNX Runtime 推理
    """

    def __init__(self, event_bus: EventBus, config: EmotionConfig):
        self.event_bus = event_bus
        self.config = config
        self.running = False

        # 模型
        self.face_emotion_model = None   # 面部表情分类器
        self.face_detector = None        # 人脸检测 (OpenCV Haar)
        self.speech_emotion_model = None # 语音情绪
        self.text_emotion_model = None   # 文本情感
        self.camera = None

        # 状态
        self.state = EmotionState()
        self.emotion_history: deque = deque(maxlen=config.emotion_history_size)
        self._last_proactive_check = 0.0

        # 情感陪伴对话模板
        self.care_prompts = self._init_care_prompts()

    async def initialize(self):
        """初始化情感计算模型"""
        logger.info("加载情感计算模型...")

        # ----- 面部表情识别 -----
        try:
            # 方案1: 使用 DeepFace (简单封装，适合原型)
            from deepface import DeepFace
            self.face_emotion_model = "deepface"
            logger.info("  面部表情识别: DeepFace")
        except ImportError:
            logger.info("  DeepFace 未安装，尝试本地模型方案...")
            try:
                # 方案2: Transformers ViT 表情分类
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                self.face_emotion_model = {
                    "processor": AutoImageProcessor.from_pretrained("trpakov/vit-face-expression"),
                    "model": AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression"),
                }
                logger.info("  面部表情识别: ViT-Face-Expression")
            except Exception:
                # 方案3: OpenCV + 简单规则（降级）
                logger.warning("  表情识别模型不可用，使用 OpenCV 基础检测")
                self.face_emotion_model = "opencv"

        # ----- 人脸检测 (OpenCV Haar Cascade — 轻量、跨平台) -----
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            logger.info("  人脸检测: OpenCV Haar Cascade")
        except Exception as e:
            logger.warning(f"  人脸检测初始化失败: {e}")

        # ----- 文本情感分析 -----
        try:
            from transformers import pipeline
            self.text_emotion_model = pipeline(
                "text-classification",
                model="BERT-wei/bert-sentiment-analysis",
                device=-1,  # CPU
            )
            logger.info("  文本情感分析: BERT Sentiment")
        except Exception:
            logger.warning("  文本情感模型不可用，使用关键词规则降级方案")
            self.text_emotion_model = "keyword"

        # ----- 摄像头 -----
        try:
            import cv2
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                logger.info("  表情识别摄像头已就绪")
            else:
                logger.warning("  摄像头不可用")
                self.camera = None
        except Exception:
            self.camera = None

        logger.info("情感计算引擎初始化完成")

    async def track_loop(self):
        """
        情绪追踪主循环 — 5 FPS (表情识别不需要高频)

        流程:
        1. 摄像头捕获面部 → 表情识别
        2. 融合历史状态 → 判断是否需要主动关怀
        3. 发布情绪事件
        """
        self.running = True
        logger.info("😊 情绪追踪已启动")

        if not self.camera:
            logger.warning("无摄像头，情绪追踪以模拟模式运行")
            await self._simulated_track_loop()
            return

        import cv2
        frame_interval = 1.0 / 5  # 5 FPS

        while self.running:
            loop_start = time.time()

            try:
                ret, frame = self.camera.read()
                if not ret:
                    await asyncio.sleep(0.5)
                    continue

                frame = cv2.flip(frame, 1)

                # 1. 人脸检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
                )

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]  # 取最大人脸
                    face_img = frame[y:y+h, x:x+w]

                    # 2. 表情识别
                    emotion, confidence = await self._recognize_face_emotion(face_img)
                    self.state.face_emotion = emotion
                    self.state.face_confidence = confidence
                else:
                    self.state.face_emotion = "neutral"
                    self.state.face_confidence = 0.0

                # 3. 融合 + 趋势分析
                await self._update_emotion_state()

                # 4. 判断是否需要主动关怀
                if self.state.needs_care:
                    now = time.time()
                    if now - self._last_proactive_check > 60:  # 每分钟最多一次
                        self._last_proactive_check = now
                        await self.event_bus.publish(Event(
                            type="emotion.detected",
                            data={
                                "emotion": self.state.dominant_emotion,
                                "confidence": self.state.fusion_confidence,
                                "face_emotion": self.state.face_emotion,
                                "trend": self.state.emotion_trend,
                                "needs_care": True,
                            }
                        ))

            except Exception as e:
                logger.error(f"情绪追踪异常: {e}")

            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)

    async def _recognize_face_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        识别面部表情

        Returns:
            (emotion_label, confidence)
        """
        try:
            if self.face_emotion_model == "deepface":
                from deepface import DeepFace
                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True,
                )
                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'].get(emotion, 0) / 100.0
                return emotion, confidence

            elif isinstance(self.face_emotion_model, dict):
                import torch
                from PIL import Image
                processor = self.face_emotion_model["processor"]
                model = self.face_emotion_model["model"]
                rgb_img = face_img[:, :, ::-1]  # BGR → RGB
                pil_img = Image.fromarray(rgb_img)
                inputs = processor(images=pil_img, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    pred_idx = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0, pred_idx].item()
                    emotion = model.config.id2label.get(pred_idx, "neutral")
                return emotion, confidence

            elif self.face_emotion_model == "opencv":
                # 降级：只能做简单的表情启发式判断
                return self._heuristic_face_emotion(face_img)

        except Exception as e:
            logger.debug(f"表情识别异常: {e}")

        return "neutral", 0.0

    def _heuristic_face_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        启发式表情判断 (无深度学习模型时的降级方案)
        基于颜色直方图、亮度等简单特征
        """
        import cv2
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # 对比度（反映表情变化程度）
        contrast = gray.std() / (gray.mean() + 1)

        # 亮度
        brightness = gray.mean()

        # 简单启发式
        if contrast > 60:
            return "surprised", 0.4
        elif brightness < 80:
            return "sad", 0.3
        elif contrast < 20:
            return "neutral", 0.5
        else:
            return "neutral", 0.4

    async def _update_emotion_state(self):
        """更新融合情绪状态 + 趋势分析"""
        # 简单多数投票融合（可改进为加权融合）
        emotions = [self.state.face_emotion]
        confidences = [self.state.face_confidence]

        if self.state.speech_confidence > 0:
            emotions.append(self.state.speech_emotion)
            confidences.append(self.state.speech_confidence)

        # 选最高置信度
        if confidences:
            best_idx = max(range(len(confidences)), key=lambda i: confidences[i])
            self.state.dominant_emotion = emotions[best_idx]
            self.state.fusion_confidence = confidences[best_idx]

        self.state.timestamp = time.time()

        # 记录历史
        self.emotion_history.append({
            "emotion": self.state.dominant_emotion,
            "confidence": self.state.fusion_confidence,
            "timestamp": self.state.timestamp,
        })

        # 趋势分析
        self._analyze_trend()

    def _analyze_trend(self):
        """分析情绪趋势：稳定 / 改善 / 恶化"""
        if len(self.emotion_history) < 10:
            return

        recent = list(self.emotion_history)[-10:]
        negative = {"sad", "angry", "fearful", "disgusted", "anxious", "tired"}
        positive = {"happy", "calm", "excited", "surprised"}

        neg_count = sum(1 for e in recent if e["emotion"] in negative)
        pos_count = sum(1 for e in recent if e["emotion"] in positive)

        # 最近一半 vs 前一半
        mid = len(recent) // 2
        recent_neg = sum(1 for e in recent[mid:] if e["emotion"] in negative)
        earlier_neg = sum(1 for e in recent[:mid] if e["emotion"] in negative)

        if recent_neg > earlier_neg + 2:
            self.state.emotion_trend = "declining"
            self.state.needs_care = True
        elif recent_neg < earlier_neg - 2:
            self.state.emotion_trend = "improving"
            self.state.needs_care = False
        else:
            self.state.emotion_trend = "stable"

        # 持续负面情绪超过 60%
        if neg_count > len(recent) * 0.6:
            self.state.needs_care = True

    async def analyze_text_sentiment(self, text: str) -> Tuple[str, float]:
        """分析文本情感"""
        try:
            if self.text_emotion_model and self.text_emotion_model != "keyword":
                result = self.text_emotion_model(text)[0]
                label = result["label"].lower()
                score = result["score"]
                # 标准化标签
                sentiment_map = {
                    "positive": "happy", "negative": "sad",
                    "neutral": "neutral",
                    "joy": "happy", "sadness": "sad",
                    "anger": "angry", "fear": "fearful",
                }
                emotion = sentiment_map.get(label, label)
                return emotion, score
        except Exception as e:
            logger.debug(f"文本情感分析失败: {e}")

        # 关键词降级方案
        emotion_keywords = {
            "happy": ["开心", "高兴", "哈哈", "太好了", "棒", "喜欢", "爱"],
            "sad": ["难过", "伤心", "哭", "失落", "遗憾", "想哭"],
            "angry": ["生气", "愤怒", "烦", "气死", "讨厌", "滚"],
            "anxious": ["紧张", "担心", "焦虑", "害怕", "不安"],
            "tired": ["累", "困", "疲劳", "没精神", "不想动"],
            "excited": ["兴奋", "激动", "太棒了", "期待"],
        }
        for emotion, keywords in emotion_keywords.items():
            if any(kw in text for kw in keywords):
                return emotion, 0.7

        return "neutral", 0.5

    async def generate_companion_response(
        self, context: dict, intent_result
    ) -> str:
        """
        生成共情陪伴回复

        根据当前情绪状态 + 用户输入，生成温暖、共情的回复
        """
        emotion = self.state.dominant_emotion
        user_text = intent_result.raw_text if intent_result else ""
        text_emotion, _ = await self.analyze_text_sentiment(user_text)

        # 优先使用文本情感
        active_emotion = text_emotion if text_emotion != "neutral" else emotion

        # 选择合适的陪伴风格
        care_style = self._select_care_style(active_emotion)

        # 生成回复（当 LLM 可用时由 LLM 生成，这里是模板）
        return self._build_care_response(active_emotion, care_style, user_text)

    def _select_care_style(self, emotion: str) -> str:
        """根据情绪选择陪伴风格"""
        style_map = {
            "happy": "celebrate",     # 一起开心，放大正面情绪
            "sad": "empathize",       # 共情安慰，温柔陪伴
            "angry": "soothe",        # 冷静安抚，转移注意力
            "anxious": "reassure",    # 鼓励肯定，增强信心
            "fearful": "protect",     # 安全感营造
            "tired": "care",          # 关怀提醒，建议休息
            "excited": "share",       # 分享喜悦
        }
        return style_map.get(emotion, "neutral")

    def _build_care_response(self, emotion: str, style: str, user_text: str) -> str:
        """构建关怀回复"""
        responses = {
            ("sad", "empathize"): [
                "我在呢。生活的路上总有些颠簸，但你不是一个人在面对。",
                "想说什么就说出来吧，我在认真听。",
                "你的感受我都懂。要不要听一首温暖的歌？",
            ],
            ("angry", "soothe"): [
                "深呼吸一下。安全到达比什么都重要，不跟那些不守规矩的人计较。",
                "我理解你现在的感受。放松，我帮你把氛围灯调成舒缓的蓝色。",
                "消消气，我放一首你最喜欢的歌吧。",
            ],
            ("anxious", "reassure"): [
                "别担心，我一直在留意周围的状况。一切都在掌控之中。",
                "你开得很好。慢慢来，我们不赶时间。",
                "紧张的时候，试试跟我一起深呼吸：吸——呼——",
            ],
            ("tired", "care"): [
                "你辛苦了。最近的服务区就在前方，要不要买杯咖啡？",
                "我注意到你好像有点累。安全最重要，休息15分钟也能让你更快到达。",
                "疲劳驾驶很危险哦。听听快节奏的音乐提提神？",
            ],
            ("fearful", "protect"): [
                "有我在，我来帮你留意路况。慢一点，安全比什么都重要。",
                "别怕，我帮你看着周围。你只需要专注前方的路。",
            ],
            ("happy", "celebrate"): [
                "看你开心我也开心！今天真是美好的一天~",
                "哈哈，心情不错嘛！保持这个状态，一路顺风！",
            ],
        }

        import random
        candidates = responses.get((emotion, style), [
            "我一直都在。无论什么时候需要我，我就在这儿。",
            "你还好吗？我随时准备好陪你聊聊天。",
        ])
        return random.choice(candidates)

    def _init_care_prompts(self):
        """初始化陪伴提示词映射"""
        return {
            "morning_greeting": "早上好！新的一天，让我们一起安全出发吧",
            "night_greeting": "晚上好！夜间驾驶请注意安全，我会一直守护着你",
            "long_drive_reminder": "你已经开了{}分钟了，休息一下吧",
            "return_greeting": "欢迎回来！感觉怎么样？",
        }

    def get_current_state(self) -> dict:
        """获取当前情绪状态"""
        return {
            "face_emotion": self.state.face_emotion,
            "face_confidence": self.state.face_confidence,
            "speech_emotion": self.state.speech_emotion,
            "text_sentiment": self.state.text_sentiment,
            "dominant_emotion": self.state.dominant_emotion,
            "emotion_trend": self.state.emotion_trend,
            "needs_care": self.state.needs_care,
        }

    async def _simulated_track_loop(self):
        """模拟情绪追踪（无摄像头时）"""
        import random
        emotions = ["neutral", "neutral", "neutral", "happy", "sad", "angry"]
        while self.running:
            await asyncio.sleep(8)
            self.state.face_emotion = random.choice(emotions)
            self.state.face_confidence = random.uniform(0.5, 0.9)
            await self._update_emotion_state()

    async def shutdown(self):
        """关闭情感计算引擎"""
        self.running = False
        if self.camera:
            self.camera.release()
        logger.info("情感计算引擎已关闭")
