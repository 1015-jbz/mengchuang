"""
意图理解引擎 — 领域分类 + 意图识别 + 槽位填充 + 多轮对话管理
基于 BERT 微调 + 规则兜底，确保端侧高效推理
"""
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from src.utils.event_bus import EventBus, Event
from configs.settings import NLUConfig


@dataclass
class IntentResult:
    """意图识别结果"""
    domain: str = "闲聊"               # 顶级领域
    intent: str = "闲聊"               # 细粒度意图
    sub_intent: Optional[str] = None   # 子意图
    slots: Dict[str, Any] = field(default_factory=dict)  # 槽位键值对
    confidence: float = 0.0            # 置信度
    raw_text: str = ""                 # 原始输入文本
    needs_clarification: bool = False  # 是否需要追问澄清

    def __repr__(self):
        return (f"IntentResult(domain={self.domain}, intent={self.intent}, "
                f"slots={self.slots}, confidence={self.confidence:.2f})")


class IntentEngine:
    """
    意图理解引擎

    技术路线:
    - Windows 原型: 基于规则 + Transformers 模型
    - LoongArch 生产: 轻量级 TextCNN/ALBERT + ONNX Runtime
    """

    def __init__(self, event_bus: EventBus, config: NLUConfig):
        self.event_bus = event_bus
        self.config = config
        self.model = None
        self.tokenizer = None
        self._intent_classifier = None
        self._slot_extractor = None

        # 对话上下文
        self.context: Dict[str, Any] = {}
        self.dialog_turns: List[Dict[str, str]] = []

    async def initialize(self):
        """加载 NLU 模型"""
        logger.info("加载意图理解模型...")

        # 规则兜底 — 总是可用
        self._init_rule_patterns()

        # 尝试加载 ML 模型
        # 默认关闭 (settings.NLUConfig.use_ml_classifier=False):
        # bert-base-chinese 的分类头是随机初始化、未经意图数据微调的，
        # 输出接近随机噪声，其“置信度”反而会覆盖规则匹配的正确结果。微调后再开启。
        self._use_ml = False
        if self.config.use_ml_classifier:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                model_name = self.config.intent_model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=len(self.DOMAIN_LABELS)
                )
                logger.info(f"  意图分类模型加载成功: {model_name}")
                self._use_ml = True
            except Exception as e:
                logger.warning(f"  ML模型加载失败 ({e})，使用规则引擎降级方案")
        else:
            logger.info("  ML 意图分类未启用（分类头未微调），使用规则引擎")

        logger.info("意图理解引擎初始化完成")

    # ============== 意图分类 ==============

    DOMAIN_LABELS = [
        "导航出行", "车辆控制", "影音娱乐", "安全守护",
        "情感陪伴", "信息查询", "闲聊"
    ]

    INTENT_HIERARCHY = {
        "导航出行": {
            "目的地搜索": ["目的地", "导航到", "去", "怎么去", "开到", "前往"],
            "途经点添加": ["途经", "路过", "顺便去"],
            "路线偏好": ["避开高速", "不走高速", "最快路线", "最短路线"],
            "路况查询": ["堵车", "路况", "前方", "拥堵"],
            "导航控制": ["取消导航", "重新规划", "放大地图"],
        },
        "车辆控制": {
            "空调控制": ["空调", "温度", "冷气", "暖气", "通风", "除雾"],
            "车窗控制": ["车窗", "窗户", "天窗", "关窗", "开窗"],
            "座椅控制": ["座椅", "座位", "加热", "通风", "按摩"],
            "灯光控制": ["大灯", "远光", "近光", "氛围灯", "阅读灯"],
            "驾驶模式": ["运动模式", "经济模式", "舒适模式", "雪地模式"],
        },
        "影音娱乐": {
            "音乐点播": ["播放", "放歌", "音乐", "来一首", "听"],
            "音量控制": ["音量", "大声", "小声", "静音"],
            "电台切换": ["FM", "AM", "电台", "收音机"],
            "播客控制": ["播客", "暂停", "继续", "下一首", "上一首"],
        },
        "安全守护": {
            "驾驶状态查询": ["开多久", "驾驶时间", "连续驾驶"],
            "车辆状态查询": ["胎压", "油量", "电量", "水温", "里程"],
            "安全提醒设置": ["疲劳提醒", "安全提醒", "提醒我"],
            "紧急求助": ["救命", "报警", "求助", "紧急", "SOS"],
        },
        "情感陪伴": {
            "主动问候": ["你好", "早", "早上好", "晚上好"],
            "情绪倾诉": ["心情不好", "不开心", "难过", "生气", "烦"],
            "趣闻闲聊": ["笑话", "讲故事", "有趣的事", "新闻"],
            "陪伴模式": ["聊天", "陪陪我", "寂寞", "无聊"],
        },
        "信息查询": {
            "天气查询": ["天气", "温度", "下雨", "刮风"],
            "车辆手册": ["怎么操作", "说明书", "怎么打开"],
            "周边搜索": ["附近", "周边", "最近的", "找个"],
            "行程管理": ["行程", "日程", "明天", "计划"],
        },
    }

    def _init_rule_patterns(self):
        """初始化规则匹配模式（编译正则表达式）"""
        # 将关键词编译为正则
        self._intent_patterns = {}
        for domain, intents in self.INTENT_HIERARCHY.items():
            for intent, keywords in intents.items():
                pattern = "|".join(re.escape(kw) for kw in keywords)
                self._intent_patterns[(domain, intent)] = re.compile(pattern)

        # 槽位提取正则
        self._slot_patterns = {
            "destination": re.compile(r"(?:去|到|导航到|前往)(.+?)(?:[，。！？\s]|$)"),
            "temperature": re.compile(r"(\d+)\s*度"),
            "volume": re.compile(r"音量\s*(\d+)"),
            "song_name": re.compile(r"(?:播放|听|来一首)(.+?)(?:的|$|[，。！？])"),
            "time_duration": re.compile(r"(\d+)\s*(?:分钟|小时)"),
        }

    async def parse(self, text: str) -> IntentResult:
        """
        解析用户输入，返回意图识别结果

        Args:
            text: 用户输入文本

        Returns:
            IntentResult: 包含领域、意图、槽位、置信度
        """
        text = text.strip()
        if not text:
            return IntentResult(raw_text=text)

        # 1. 意图分类
        domain, intent, confidence = self._classify_intent(text)

        # 2. 槽位填充
        slots = self._extract_slots(text, domain, intent)

        # 3. 置信度不足时标记澄清
        needs_clarification = confidence < self.config.confidence_threshold

        result = IntentResult(
            domain=domain,
            intent=intent,
            slots=slots,
            confidence=confidence,
            raw_text=text,
            needs_clarification=needs_clarification,
        )

        logger.debug(f"NLU: {text} → {result}")
        return result

    def _classify_intent(self, text: str) -> Tuple[str, str, float]:
        """
        意图分类

        1. 先用规则匹配（快速、准确）
        2. 规则不确定时用 ML 模型
        3. 都不匹配返回"闲聊"
        """
        best_domain = "闲聊"
        best_intent = "闲聊"
        best_score = 0.0

        # 规则匹配
        for (domain, intent), pattern in self._intent_patterns.items():
            match = pattern.search(text)
            if match:
                # 计算粗略置信度（匹配长度/文本长度）
                score = len(match.group()) / len(text) if text else 0
                # 加长关键词匹配加分
                score = min(score + 0.3, 1.0)
                if score > best_score:
                    best_score = score
                    best_domain = domain
                    best_intent = intent

        # 规则命中且置信度够高，直接返回
        if best_score > 0.5:
            return best_domain, best_intent, best_score

        # 规则没命中但可能命中领域关键词，用 ML 模型
        if self._use_ml and self.model and self.tokenizer:
            try:
                import torch
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=128
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    predicted_idx = torch.argmax(probs, dim=-1).item()
                    ml_confidence = probs[0, predicted_idx].item()

                    if ml_confidence > best_score:
                        best_domain = self.DOMAIN_LABELS[predicted_idx]
                        best_intent = "通用"
                        best_score = ml_confidence
            except Exception as e:
                logger.debug(f"ML 分类失败: {e}")

        return best_domain, best_intent, best_score

    def _extract_slots(self, text: str, domain: str, intent: str) -> Dict[str, Any]:
        """槽位填充 — 提取意图中的关键参数"""
        slots = {}

        # 通用槽位
        for slot_name, pattern in self._slot_patterns.items():
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                if slot_name == "temperature":
                    slots[slot_name] = int(value)
                elif slot_name == "volume":
                    slots[slot_name] = int(value)
                else:
                    slots[slot_name] = value

        # 领域特定槽位
        if domain == "车辆控制" and intent == "空调控制":
            if "temperature" not in slots:
                temp_match = re.search(r"(\d+)\s*度", text)
                if temp_match:
                    slots["temperature"] = int(temp_match.group(1))
            slots["action"] = "on" if any(w in text for w in ["开", "打开", "启动"]) else \
                              "off" if any(w in text for w in ["关", "关闭"]) else "toggle"

        elif domain == "导航出行":
            dest_match = re.search(r"(?:去|到|导航到|前往)(?P<dest>.+?)(?:[，。！？\s]|$)", text)
            if dest_match:
                slots["destination"] = dest_match.group("dest").strip()

        elif domain == "影音娱乐" and intent == "音乐点播":
            song_match = re.search(r"(?:播放|听|来一首)\s*(?P<song>.+?)(?:的(?:歌|音乐)|$|[，。！？])", text)
            if song_match:
                slots["song_name"] = song_match.group("song").strip()

        return slots

    # ============== 多轮对话管理 ==============

    def update_context(self, user_text: str, intent_result: IntentResult, system_response: str):
        """更新对话上下文"""
        self.dialog_turns.append({
            "user": user_text,
            "intent": f"{intent_result.domain}/{intent_result.intent}",
            "system": system_response,
        })
        # 滑动窗口
        if len(self.dialog_turns) > self.config.max_context_turns:
            self.dialog_turns = self.dialog_turns[-self.config.max_context_turns:]

    def get_dialog_context(self) -> str:
        """获取对话上下文（用于 LLM prompt）"""
        if not self.dialog_turns:
            return ""
        lines = []
        for turn in self.dialog_turns[-3:]:  # 最近3轮
            lines.append(f"用户: {turn['user']}")
            lines.append(f"系统: {turn['system']}")
        return "\n".join(lines)
