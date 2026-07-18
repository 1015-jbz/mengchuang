"""
集成测试 — 验证 CLI 架构链路: 事件总线 → 编排器 → 意图理解 → 路由 → 回复

覆盖: speech.recognized 事件注入 → NLU → _route_intent → 各域处理器 → speak
不依赖: 麦克风 / 大模型 / 网络（speech.speak 被替换为收集器，不真正播音）

运行: python tests/test_pipeline.py   （在项目根目录）
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.event_bus import EventBus, Event
from src.orchestrator import CockpitOrchestrator

# (输入文本, 预期领域, 回复中应包含的关键词)
CASES = [
    ("导航到天安门",   "导航出行", "天安门"),
    ("打开空调26度",   "车辆控制", "26"),
    ("播放周杰伦的歌", "影音娱乐", "周杰伦"),
    ("胎压正常吗",     "安全守护", "正常"),
    ("我今天心情不好", "情感陪伴", ""),      # 共情回复内容不固定，只要有回复即可
    ("给我讲讲你自己", "闲聊",     ""),      # LLM 模板回复
]


async def main():
    bus = EventBus()
    orch = CockpitOrchestrator(bus)
    await orch.initialize()

    # 替换 speak 为收集器（不播音、不依赖 TTS）
    spoken = []

    async def fake_speak(text, interrupt=False):
        spoken.append(text)

    orch.speech.speak = fake_speak

    # 启动事件总线分发循环
    dispatcher = asyncio.create_task(bus.start())

    failed = []
    for text, domain, expect in CASES:
        spoken.clear()
        await bus.publish(Event(
            type="speech.recognized",
            data={"text": text, "speaker": "driver", "source": "test"},
        ))
        await asyncio.sleep(0.8)  # 等待分发 + 处理

        reply = spoken[-1] if spoken else ""
        ok = bool(reply) and (expect in reply if expect else True)
        if not ok:
            failed.append(text)
        print(f"[{'PASS' if ok else 'FAIL'}] [{domain}] {text!r:22} -> {reply or '(无回复)'}")

    bus.stop()
    dispatcher.cancel()
    await orch.shutdown()

    print("\n结果:", "全部通过" if not failed else f"失败 {len(failed)} 条: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
