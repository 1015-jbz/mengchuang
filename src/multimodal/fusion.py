"""
多模态融合引擎 — 融合语音/视觉/车辆信号，构建统一上下文
"""
from typing import Dict, Any
from loguru import logger
from src.utils.event_bus import EventBus


class MultimodalFusion:
    """
    多模态融合引擎

    融合维度:
    - 语音语义 + 面部表情 → 真实意图理解
    - 疲劳检测 + 驾驶时长 → 综合风险评分
    - 情绪状态 + 驾驶场景 → 个性化响应策略
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.current_context: Dict[str, Any] = {}
        self.fusion_weights = {
            "speech": 0.4,    # 语音权重
            "vision": 0.35,   # 视觉权重
            "vehicle": 0.25,  # 车辆信号权重
        }

    async def initialize(self):
        logger.info("多模态融合引擎初始化完成")

    async def build_context(
        self,
        text: str,
        intent,
        safety: dict,
        emotion: dict,
    ) -> Dict[str, Any]:
        """
        构建多模态融合上下文

        综合语音意图 + 安全状态 + 情绪状态 + 驾驶场景
        """
        context = {
            # 语音层
            "text": text,
            "intent": {
                "domain": intent.domain,
                "intent": intent.intent,
                "slots": intent.slots,
                "confidence": intent.confidence,
            },

            # 安全层
            "safety": safety,

            # 情绪层
            "emotion": emotion,

            # 融合决策
            "fusion": self._compute_fusion_decision(intent, safety, emotion),

            # 对话上下文标记
            "is_urgent": safety.get("fatigue_level", "normal") in ("high", "critical"),
            "is_emotional": emotion.get("dominant_emotion", "neutral") != "neutral",
            "needs_care": emotion.get("needs_care", False),
            "driving_focus_required": safety.get("is_distracted", False),
        }

        # 多轮对话上下文
        context["conversation_context"] = self._format_for_llm(context)
        self.current_context = context
        return context

    def _compute_fusion_decision(self, intent, safety: dict, emotion: dict) -> dict:
        """
        多模态融合决策

        Returns:
            {
                "priority": "safety" | "emotion" | "normal",
                "response_style": "urgent" | "caring" | "professional" | "casual",
                "should_interrupt": bool,
            }
        """
        fatigue = safety.get("fatigue_level", "normal")
        dominant_emotion = emotion.get("dominant_emotion", "neutral")

        # 安全优先
        if fatigue in ("high", "critical"):
            return {
                "priority": "safety",
                "response_style": "urgent",
                "should_interrupt": True,
            }

        # 情感优先
        if emotion.get("needs_care", False) and dominant_emotion in (
            "sad", "angry", "anxious", "fearful"
        ):
            return {
                "priority": "emotion",
                "response_style": "caring",
                "should_interrupt": False,
            }

        # 正常模式
        return {
            "priority": "normal",
            "response_style": "professional",
            "should_interrupt": False,
        }

    def _format_for_llm(self, context: dict) -> str:
        """将融合上下文格式化为 LLM 可理解的文本"""
        parts = []

        safety = context.get("safety", {})
        if safety.get("fatigue_level", "normal") != "normal":
            parts.append(f"[驾驶状态] 疲劳等级: {safety['fatigue_level']}")

        emotion = context.get("emotion", {})
        if emotion.get("dominant_emotion", "neutral") != "neutral":
            emo_zh = {
                "happy": "开心", "sad": "悲伤", "angry": "愤怒",
                "anxious": "焦虑", "tired": "疲倦", "fearful": "恐惧",
            }.get(emotion["dominant_emotion"], emotion["dominant_emotion"])
            parts.append(f"[驾驶员情绪] {emo_zh}")

        intent = context.get("intent", {})
        if intent.get("domain"):
            parts.append(f"[意图] {intent['domain']}/{intent['intent']}")

        return " | ".join(parts) if parts else ""

    async def update_safety_alert(self, alert_data: dict):
        """更新安全告警到融合上下文"""
        self.current_context["latest_alert"] = alert_data

    async def shutdown(self):
        pass
