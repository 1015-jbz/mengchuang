"""
智能座舱多模态交互终端 — 主入口
"""
import sys
import asyncio
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger
from src.utils.event_bus import EventBus
from src.orchestrator import CockpitOrchestrator


def setup_logging():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/smart_cockpit_{time:YYYY-MM-DD}.log",
        rotation="100 MB",
        retention="7 days",
        level="DEBUG"
    )


async def main():
    """主函数"""
    setup_logging()
    logger.info("=" * 60)
    logger.info("🚗 智能座舱多模态交互终端启动中...")
    logger.info("   基于 LoongArch 端侧 AI")
    logger.info("=" * 60)

    # 初始化事件总线
    event_bus = EventBus()

    # 初始化编排器
    orchestrator = CockpitOrchestrator(event_bus)

    try:
        # 初始化所有模块
        await orchestrator.initialize()

        logger.info("✅ 所有模块初始化完成")
        logger.info("💡 唤醒词: '你好小航'")
        logger.info("💡 按 Ctrl+C 退出")

        # 启动主循环
        await orchestrator.run()

    except KeyboardInterrupt:
        logger.info("🛑 收到退出信号")
    except Exception as e:
        logger.exception(f"❌ 运行异常: {e}")
    finally:
        await orchestrator.shutdown()
        logger.info("👋 智能座舱系统已关闭")


if __name__ == "__main__":
    asyncio.run(main())
