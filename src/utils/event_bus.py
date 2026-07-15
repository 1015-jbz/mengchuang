"""
事件总线 — 模块间松耦合通信
基于 asyncio.Queue 的发布-订阅模式
"""
import asyncio
from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Event:
    """事件"""
    type: str                       # 事件类型
    data: Dict[str, Any] = field(default_factory=dict)  # 事件数据
    source: str = ""                # 事件来源模块


class EventBus:
    """
    异步事件总线

    使用场景:
    - speech.recognized → Orchestrator → LLM 推理
    - safety.alert → Orchestrator → TTS 语音告警
    - emotion.detected → Orchestrator → 主动关怀
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    def subscribe(self, event_type: str, callback: Callable):
        """订阅事件"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.debug(f"订阅事件: {event_type} → {callback.__name__}")

    def unsubscribe(self, event_type: str, callback: Callable):
        """取消订阅"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)

    async def publish(self, event: Event):
        """发布事件（异步）"""
        await self._queue.put(event)

    def publish_sync(self, event: Event):
        """发布事件（同步，用于非异步上下文）"""
        self._queue.put_nowait(event)

    async def start(self):
        """启动事件分发循环"""
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"事件分发异常: {e}")

    async def _dispatch(self, event: Event):
        """分发事件给订阅者"""
        subscribers = self._subscribers.get(event.type, [])
        if not subscribers:
            logger.debug(f"事件无订阅者: {event.type}")
            return

        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"事件回调异常 [{callback.__name__}]: {e}")

    def stop(self):
        """停止事件总线"""
        self._running = False
