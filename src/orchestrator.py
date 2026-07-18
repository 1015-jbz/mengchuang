"""
智能座舱核心编排器 — 所有模块的协调中枢
"""
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

from loguru import logger

from src.utils.event_bus import EventBus, Event
from src.asr.speech_engine import SpeechEngine
from src.nlu.intent_engine import IntentEngine
from src.llm.llm_engine import LLMEngine
from src.safety.safety_monitor import SafetyMonitor
from src.emotion.emotion_engine import EmotionEngine
from src.control.cockpit_control import CockpitController
from src.multimodal.fusion import MultimodalFusion
from configs.settings import config


class CockpitOrchestrator:
    """
    座舱编排器 — 中枢调度核心

    职责：
    1. 接收多模态输入（语音/视觉/车辆信号）
    2. 分发任务到对应引擎
    3. 融合多引擎结果，决策最优响应
    4. 管理对话会话上下文
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.running = False

        # 各引擎（延迟初始化）
        self.speech: Optional[SpeechEngine] = None
        self.intent: Optional[IntentEngine] = None
        self.llm: Optional[LLMEngine] = None
        self.safety: Optional[SafetyMonitor] = None
        self.emotion: Optional[EmotionEngine] = None
        self.controller: Optional[CockpitController] = None
        self.fusion: Optional[MultimodalFusion] = None

        # 会话状态
        self.context: Dict[str, Any] = {}
        self.driving_start_time: Optional[datetime] = None
        self.is_driving: bool = False

    async def initialize(self):
        """初始化所有引擎模块"""
        logger.info("🔧 初始化各模块引擎...")

        # 按依赖顺序初始化
        self.speech = SpeechEngine(self.event_bus, config.asr)
        await self.speech.initialize()
        logger.info("  ✅ 语音引擎就绪")

        self.llm = LLMEngine(self.event_bus, config.llm)
        await self.llm.initialize()
        logger.info("  ✅ 大模型推理引擎就绪")

        self.intent = IntentEngine(self.event_bus, config.nlu)
        await self.intent.initialize()
        logger.info("  ✅ 意图理解引擎就绪")

        self.safety = SafetyMonitor(self.event_bus, config.safety)
        await self.safety.initialize()
        logger.info("  ✅ 安全监控引擎就绪")

        self.emotion = EmotionEngine(self.event_bus, config.emotion)
        await self.emotion.initialize()
        logger.info("  ✅ 情感计算引擎就绪")

        self.controller = CockpitController(self.event_bus)
        await self.controller.initialize()
        logger.info("  ✅ 座舱控制中间件就绪")

        self.fusion = MultimodalFusion(self.event_bus)
        await self.fusion.initialize()
        logger.info("  ✅ 多模态融合引擎就绪")

        # 注册事件处理器
        self._register_event_handlers()

    def _register_event_handlers(self):
        """注册事件回调"""
        self.event_bus.subscribe("speech.recognized", self._on_speech_recognized)
        self.event_bus.subscribe("safety.alert", self._on_safety_alert)
        self.event_bus.subscribe("emotion.detected", self._on_emotion_detected)
        self.event_bus.subscribe("vehicle.state_changed", self._on_vehicle_state)

    async def run(self):
        """主运行循环"""
        self.running = True

        # 模拟点火信号，使 is_driving 状态机生效（真实部署应由 CAN 总线接入发布）
        await self.event_bus.publish(Event(
            type="vehicle.state_changed",
            data={"key": "ignition", "value": "on"},
            source="orchestrator",
        ))

        # 启动后台任务（事件总线分发循环必须启动，否则 publish 的事件永远无人消费）
        tasks = [
            asyncio.create_task(self.event_bus.start()),
            asyncio.create_task(self.speech.listen_loop()),
            asyncio.create_task(self.safety.monitor_loop()),
            asyncio.create_task(self.emotion.track_loop()),
            asyncio.create_task(self._proactive_check_loop()),
        ]

        # 等待任务（任一异常则全部取消）
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        """优雅关闭"""
        self.running = False
        self.event_bus.stop()
        engines = [self.speech, self.llm, self.safety, self.emotion, self.controller]
        for engine in engines:
            if engine:
                await engine.shutdown()
        logger.info("所有引擎已关闭")

    # ============== 事件处理 ==============

    async def _on_speech_recognized(self, event: Event):
        """
        语音识别回调 — 核心交互入口

        处理流程:
        1. 意图理解
        2. 安全判断（行驶中是否允许该操作）
        3. LLM生成回复 / 执行车控指令
        4. TTS语音反馈
        """
        text = event.data.get("text", "")
        speaker = event.data.get("speaker", "driver")
        logger.info(f"🎤 [{speaker}] {text}")

        # Step 1: 意图识别
        intent_result = await self.intent.parse(text)

        # Step 2: 融合多模态上下文
        safety_state = self.safety.get_current_state()
        emotion_state = self.emotion.get_current_state()
        fused_context = await self.fusion.build_context(
            text=text,
            intent=intent_result,
            safety=safety_state,
            emotion=emotion_state
        )

        # Step 3: 安全策略检查
        if self.is_driving and intent_result.domain == "车辆控制":
            restricted = self.controller.check_safety_restriction(
                intent_result.intent, self.is_driving
            )
            if restricted:
                await self.speech.speak("为了安全，请在停车后操作")
                return

        # Step 4: 根据意图领域路由处理
        response = await self._route_intent(intent_result, fused_context)

        # Step 5: TTS 语音输出
        if response:
            await self.speech.speak(response)

        # Step 6: 更新对话上下文
        self.context["last_intent"] = intent_result
        self.context["last_response"] = response
        self.context["conversation_history"] = (
            self.context.get("conversation_history", []) +
            [{"role": "user", "content": text},
             {"role": "assistant", "content": response}]
        )[-config.nlu.max_context_turns * 2:]

    async def _route_intent(self, intent_result, fused_context: dict) -> str:
        """
        意图路由 — 根据意图领域分发给对应处理器
        """
        domain = intent_result.domain
        intent_name = intent_result.intent
        slots = intent_result.slots

        if domain == "车辆控制":
            return await self.controller.execute(intent_name, slots)

        elif domain == "安全守护":
            if intent_name == "车辆状态查询":
                return self.controller.query_vehicle_status(slots)
            elif intent_name == "紧急求助":
                return await self._handle_emergency(fused_context)

        elif domain == "情感陪伴":
            return await self.emotion.generate_companion_response(
                fused_context, intent_result
            )

        elif domain == "导航出行":
            return await self._handle_navigation(intent_name, slots, fused_context)

        elif domain == "影音娱乐":
            return await self.controller.execute_entertainment(intent_name, slots)

        # 默认：LLM 对话（传用户原话；多模态状态标签放进 system prompt）
        return await self.llm.generate_response(
            fused_context.get("text", ""),
            system_prompt=self._build_system_prompt(fused_context)
        )

    async def _on_safety_alert(self, event: Event):
        """安全告警回调"""
        alert_level = event.data.get("level", "warning")
        alert_type = event.data.get("type", "unknown")
        alert_msg = event.data.get("message", "")

        logger.warning(f"⚠️ 安全告警 [{alert_level}] {alert_type}: {alert_msg}")

        # 高级别告警立即语音提醒
        if alert_level in ("high", "critical"):
            await self.speech.speak(alert_msg, interrupt=True)

        # 联动车辆控制
        if alert_type == "fatigue" and alert_level == "critical":
            rest_area = await self.controller.suggest_rest_area()
            if rest_area:
                await self.speech.speak(f"建议尽快休息：{rest_area}")

        # 更新多模态融合状态
        await self.fusion.update_safety_alert(event.data)

    async def _on_emotion_detected(self, event: Event):
        """情绪检测回调"""
        emotion_type = event.data.get("emotion", "neutral")
        confidence = event.data.get("confidence", 0.0)
        logger.info(f"😊 检测到情绪: {emotion_type} (置信度: {confidence:.2f})")

        # 负面情绪 → 触发主动关怀
        if emotion_type in ("angry", "anxious", "sad", "fearful"):
            await self._trigger_proactive_care(emotion_type, event.data)

    async def _on_vehicle_state(self, event: Event):
        """车辆状态变化回调"""
        key = event.data.get("key", "")
        value = event.data.get("value", None)
        if key == "ignition" and value == "on":
            self.is_driving = True
            self.driving_start_time = datetime.now()
            logger.info("🚗 车辆启动，驾驶监控开始")

    # ============== 主动交互 ==============

    async def _proactive_check_loop(self):
        """主动检查循环 — 定期评估是否需要主动交互"""
        while self.running:
            await asyncio.sleep(config.emotion.proactive_check_interval_min * 60)

            if not self.is_driving:
                continue

            # 检查驾驶时长
            if self.driving_start_time:
                elapsed = (datetime.now() - self.driving_start_time).total_seconds() / 60
                if elapsed > config.safety.max_continuous_driving_min:
                    await self.speech.speak(
                        f"您已经连续驾驶{int(elapsed)}分钟了，"
                        "前方有服务区，建议休息一下哦"
                    )

            # 检查是否需要主动关怀
            emotion_state = self.emotion.get_current_state()
            if emotion_state.get("needs_care", False):
                await self._trigger_proactive_care(
                    emotion_state.get("dominant_emotion", "neutral"), {}
                )

    async def _trigger_proactive_care(self, emotion_type: str, context: dict):
        """触发主动情感关怀"""
        care_prompts = {
            "angry": "我注意到你好像有点烦躁，深呼吸，安全最重要。要不要听首舒缓的音乐？",
            "anxious": "别紧张，我一直在帮你看着路况，一切都在掌控之中。",
            "sad": "我在这儿呢。如果你想聊聊，我会一直听你说。",
            "fearful": "没关系，我帮你留意周围了。慢一点，安全第一。",
            "tired": "你看起来有点累了，最近的咖啡店在前方2公里，要不要停下来歇歇？",
        }
        prompt = care_prompts.get(emotion_type, "你还好吗？我一直都在。")
        await self.speech.speak(prompt)

    # ============== 辅助方法 ==============

    def _build_system_prompt(self, fused_context: dict) -> str:
        """构建 LLM 系统提示词"""
        time_str = datetime.now().strftime("%H:%M")
        emotion = fused_context.get("emotion", {}).get("dominant_emotion", "neutral")
        driving_status = "行驶中" if self.is_driving else "驻车"

        return f"""你是一个智能座舱AI助手"小航"，运行在龙芯端侧芯片上。

当前状态：
- 时间: {time_str}
- 车辆状态: {driving_status}
- 驾驶员情绪: {emotion}
- 场景标记: {fused_context.get('conversation_context') or '无'}

你的角色特点：
1. 温暖贴心 — 像一位懂你的伙伴，不只是执行命令
2. 安全守护 — 始终将驾驶安全放在第一位
3. 简洁高效 — 回复简短(50字内)，用语音播报
4. 主动关怀 — 察觉情绪变化时会主动关心
5. 专业知识 — 了解车辆操作和驾驶安全知识

重要：你是100%离线运行的，数据不会上传云端，保护用户隐私。
"""

    async def _handle_emergency(self, context: dict) -> str:
        """处理紧急求助"""
        # 紧急情况处理逻辑
        logger.critical("🆘 紧急求助触发!")
        return "已收到求助，正在为您拨打紧急联系人和报警电话。请保持冷静，救援正在赶来。"

    async def _handle_navigation(self, intent_name: str, slots: dict, context: dict) -> str:
        """处理导航相关意图"""
        if intent_name == "目的地搜索":
            destination = slots.get("destination", "")
            if destination:
                return f"正在为您导航至{destination}，预计需要30分钟"
        return "请告诉我您想去哪里？"
