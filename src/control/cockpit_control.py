"""
座舱控制中间件 — 自然语言指令 → 车辆控制信号
"""
import random
from typing import Dict, Any, Optional
from loguru import logger
from src.utils.event_bus import EventBus, Event


class CockpitController:
    """
    座舱控制中间件

    职责:
    1. NL指令 → CAN信号映射（当前为模拟）
    2. 安全权限检查（行驶中禁用部分功能）
    3. 场景模式管理
    """

    # 行驶中禁止的操作
    RESTRICTED_WHILE_DRIVING = [
        "视频播放", "游戏", "座椅按摩",
        "天窗全开", "车窗全开",
    ]

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.device_states: Dict[str, Any] = {
            "ac": {"power": "off", "temperature": 22, "mode": "auto"},
            "windows": {"front_left": "closed", "front_right": "closed",
                        "rear_left": "closed", "rear_right": "closed"},
            "sunroof": "closed",
            "seat_heat": {"driver": "off", "passenger": "off"},
            "ambient_light": {"power": "off", "color": "blue", "brightness": 50},
            "audio": {"power": "off", "volume": 30, "source": "bluetooth"},
        }

    async def initialize(self):
        logger.info("座舱控制中间件初始化完成")

    def check_safety_restriction(self, intent: str, is_driving: bool) -> bool:
        """
        检查安全限制

        Returns:
            True = 受限，False = 允许
        """
        if not is_driving:
            return False
        for restricted in self.RESTRICTED_WHILE_DRIVING:
            if restricted in intent:
                return True
        return False

    async def execute(self, intent: str, slots: dict) -> str:
        """执行车控指令"""
        response = ""

        if "空调" in intent:
            response = self._control_ac(slots)
        elif "车窗" in intent or "天窗" in intent:
            response = self._control_windows(intent, slots)
        elif "座椅" in intent:
            response = self._control_seat(intent, slots)
        elif "灯光" in intent:
            response = self._control_light(intent, slots)
        elif "驾驶模式" in intent:
            response = self._control_drive_mode(intent, slots)
        else:
            response = f"好的，已执行{intent}操作"

        # 发布状态变更（事件名须与 orchestrator 的订阅一致；Event 是模块级类，不是 EventBus 属性）
        await self.event_bus.publish(Event(
            type="vehicle.state_changed",
            data={"intent": intent, "slots": slots, "response": response},
            source="cockpit_control",
        ))

        return response

    def _control_ac(self, slots: dict) -> str:
        ac = self.device_states["ac"]
        if "temperature" in slots:
            temp = slots["temperature"]
            ac["temperature"] = temp
            ac["power"] = "on"
            return f"好的，空调温度已设为{temp}°C"
        action = slots.get("action", "toggle")
        if action == "on":
            ac["power"] = "on"
            return "空调已打开"
        elif action == "off":
            ac["power"] = "off"
            return "空调已关闭"
        return "空调状态已更新"

    def _control_windows(self, intent: str, slots: dict) -> str:
        if "天窗" in intent:
            self.device_states["sunroof"] = "open"
            return "天窗已打开"
        return "车窗已操作"

    def _control_seat(self, intent: str, slots: dict) -> str:
        if "加热" in intent:
            self.device_states["seat_heat"]["driver"] = "on"
            return "座椅加热已开启"
        return "座椅设置已更新"

    def _control_light(self, intent: str, slots: dict) -> str:
        self.device_states["ambient_light"]["power"] = "on"
        return "氛围灯已打开"

    def _control_drive_mode(self, intent: str, slots: dict) -> str:
        mode_map = {"运动": "sport", "经济": "eco", "舒适": "comfort"}
        for name, mode in mode_map.items():
            if name in intent:
                return f"驾驶模式已切换为{name}模式"
        return "驾驶模式已切换"

    def query_vehicle_status(self, slots: dict) -> str:
        """查询车辆状态"""
        checks = {
            "胎压": f"胎压正常，四轮均在2.3-2.5 bar范围内",
            "油量": "油量剩余65%，可行驶约400公里",
            "电量": "动力电池电量82%，续航约350公里",
            "水温": "发动机水温正常，90°C",
            "总里程": "总里程38,520公里",
        }
        for key, msg in checks.items():
            if key in str(slots):
                return msg
        return "车辆各系统状态正常"

    async def execute_entertainment(self, intent: str, slots: dict) -> str:
        """娱乐系统控制（async 与 execute 保持一致，orchestrator 中以 await 调用）"""
        if "音量" in intent:
            vol = slots.get("volume", 30)
            self.device_states["audio"]["volume"] = vol
            return f"音量已调至{vol}"
        if "播放" in intent or "点播" in intent:
            song = slots.get("song_name", "")
            if song:
                return f"正在为您播放：{song}"
            return "好的，为您播放音乐"
        if "暂停" in intent:
            return "已暂停播放"
        if "下一首" in intent:
            return "下一首"
        return "好的"

    async def suggest_rest_area(self):
        """建议休息区"""
        logger.info("🏪 搜索最近服务区...")
        # 模拟：最近的休息区
        rest_areas = [
            "前方3公里 — 太阳岛服务区",
            "前方8公里 — 龙岗服务区",
            "前方15公里 — 碧海服务区",
        ]
        return random.choice(rest_areas)

    async def shutdown(self):
        pass
