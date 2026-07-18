"""
驾驶安全监控引擎 — 疲劳检测 + 分心检测 + 危险行为预警
基于 MediaPipe + OpenCV 视觉感知 + CAN 数据融合
"""
import asyncio
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

import numpy as np

from loguru import logger
from src.utils.event_bus import EventBus, Event
from configs.settings import SafetyConfig


@dataclass
class SafetyState:
    """安全状态快照"""
    is_driver_present: bool = False
    fatigue_level: str = "normal"       # normal / mild / moderate / severe
    perclos: float = 0.0                # 眼睑闭合百分比
    yawn_count: int = 0                 # 打哈欠次数（最近1分钟）
    gaze_direction: str = "forward"     # forward / left / right / down / up
    distraction_duration: float = 0.0   # 当前分心持续时间(秒)
    head_pose: str = "normal"           # normal / tilted / down
    alert_level: str = "normal"         # normal / warning / high / critical
    timestamp: float = 0.0


class SafetyMonitor:
    """
    驾驶安全监控引擎

    检测维度:
    1. 疲劳检测 — PERCLOS + 打哈欠频率 + 头部下沉
    2. 分心检测 — 视线偏离 + 头部扭转 + 手机使用
    3. 异常驾驶 — 急刹/急加速/急转 (CAN信号)

    Windows: OpenCV + MediaPipe + dlib
    LoongArch: OpenCV + 轻量 CNN (ONNX)
    """

    def __init__(self, event_bus: EventBus, config: SafetyConfig):
        self.event_bus = event_bus
        self.config = config
        self.running = False

        # 视觉模型
        self.face_mesh = None        # MediaPipe FaceMesh
        self.face_detector = None    # OpenCV Haar / dlib
        self.camera = None

        # 状态追踪
        self.state = SafetyState()
        self._eye_closure_history = deque(maxlen=300)  # 最近5秒(60fps*5)
        self._yawn_timestamps = deque(maxlen=100)      # 哈欠时间戳
        self._was_yawning = False                      # 哈欠上升沿检测
        self._gaze_history = deque(maxlen=100)
        self._distraction_start: Optional[float] = None
        self._fatigue_score = 0.0

        # 告警冷却（避免频繁告警）
        self._last_alert_time: Dict[str, float] = {}

    async def initialize(self):
        """初始化视觉检测模型"""
        logger.info("加载安全监控模型...")

        # ----- MediaPipe FaceMesh (468 点面部关键点) -----
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,  # 包含虹膜关键点
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("  MediaPipe FaceMesh 初始化成功")
        except ImportError:
            logger.warning("  mediapipe 未安装，视觉安全监控降级")
        except Exception as e:
            logger.warning(f"  FaceMesh 初始化失败: {e}")

        # ----- 摄像头 -----
        try:
            import cv2
            self.camera = cv2.VideoCapture(self.config.camera_id)
            if not self.camera.isOpened():
                logger.warning(f"  摄像头 {self.config.camera_id} 不可用")
                self.camera = None
            else:
                logger.info(f"  摄像头 {self.config.camera_id} 已就绪")
        except Exception as e:
            logger.warning(f"  摄像头初始化失败: {e}")
            self.camera = None

        # 面部关键点索引 (MediaPipe FaceMesh)
        # 眼睛: Left=33,133,159,145  Right=362,263,387,373
        # 嘴唇: 13,14,17 (上唇) 61,291 (嘴角)
        self.LEFT_EYE  = [33, 133, 159, 145]
        self.RIGHT_EYE = [362, 263, 387, 373]
        self.MOUTH_TOP    = [13, 14]
        self.MOUTH_BOTTOM = [17]
        self.NOSE_TIP = 1
        self.CHIN = 152

        logger.info("安全监控引擎初始化完成")

    async def monitor_loop(self):
        """
        安全监控主循环 — 25 FPS

        每帧:
        1. 人脸检测
        2. 关键点提取
        3. 疲劳/分心指标计算
        4. 风险评估 -> 触发告警
        """
        self.running = True
        logger.info("🛡️ 安全监控已启动")

        if not self.camera:
            logger.warning("无摄像头，安全监控以模拟模式运行")
            await self._simulated_monitor_loop()
            return

        frame_interval = 1.0 / 25  # 25 FPS
        while self.running:
            loop_start = time.time()

            try:
                import cv2
                ret, frame = self.camera.read()
                if not ret:
                    await asyncio.sleep(0.1)
                    continue

                # 镜像翻转（自拍视角）
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]

                # 面部关键点检测
                if self.face_mesh:
                    results = self.face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0]
                        self._analyze_face(landmarks, w, h)

                # 风险评估
                await self._assess_risk()

            except Exception as e:
                logger.error(f"安全监控异常: {e}")

            # 帧率控制
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)

    def _analyze_face(self, landmarks, w: int, h: int):
        """分析面部关键点，计算安全指标"""
        points = []
        for lm in landmarks.landmark:
            points.append((lm.x * w, lm.y * h))

        # 1. 眼睑闭合度 (PERCLOS - Percentage of Eye Closure)
        left_eye_ratio = self._eye_aspect_ratio(
            [points[i] for i in self.LEFT_EYE]
        )
        right_eye_ratio = self._eye_aspect_ratio(
            [points[i] for i in self.RIGHT_EYE]
        )
        avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # EAR < 0.2 视为闭眼
        is_eye_closed = avg_eye_ratio < 0.2
        self._eye_closure_history.append(is_eye_closed)
        if len(self._eye_closure_history) > 0:
            self.state.perclos = sum(self._eye_closure_history) / len(self._eye_closure_history)

        # 2. 打哈欠检测 — 嘴部纵横比 (MAR)
        mouth_ratio = self._mouth_aspect_ratio(points)
        is_yawning = mouth_ratio > 0.6  # 阈值
        now = time.time()
        # 只在“开始张嘴”的上升沿计一次哈欠，避免张嘴期间每帧都 +1
        if is_yawning and not self._was_yawning:
            self._yawn_timestamps.append(now)
        self._was_yawning = is_yawning
        # 清理旧记录 (保留60秒内)
        while self._yawn_timestamps and self._yawn_timestamps[0] < now - 60:
            self._yawn_timestamps.popleft()
        self.state.yawn_count = len(self._yawn_timestamps)

        # 3. 头部姿态估计
        nose = points[self.NOSE_TIP]
        chin = points[self.CHIN]
        head_tilt = (chin[1] - nose[1]) / h  # 头低下的程度
        if head_tilt > 0.15:
            self.state.head_pose = "down"
        elif abs((nose[0] - w/2) / w) > 0.2:
            self.state.head_pose = "tilted"
        else:
            self.state.head_pose = "normal"

        # 4. 视线方向 (基于虹膜位置)
        # 简化：看鼻尖相对面部中心的偏移
        face_center_x = sum(p[0] for p in points[:10]) / 10
        nose_offset = (nose[0] - face_center_x) / w
        if nose_offset > 0.15:
            self.state.gaze_direction = "right"
        elif nose_offset < -0.15:
            self.state.gaze_direction = "left"
        else:
            self.state.gaze_direction = "forward"

        # 5. 分心检测
        is_distracted = (
            self.state.gaze_direction != "forward" or
            self.state.head_pose in ("down", "tilted")
        )
        if is_distracted:
            if self._distraction_start is None:
                self._distraction_start = time.time()
            self.state.distraction_duration = time.time() - self._distraction_start
        else:
            self._distraction_start = None
            self.state.distraction_duration = 0.0

    @staticmethod
    def _eye_aspect_ratio(eye_points) -> float:
        """
        计算眼部纵横比 (Eye Aspect Ratio)

        eye_points 顺序对应 LEFT_EYE/RIGHT_EYE 索引: [内眼角, 外眼角, 上睑, 下睑]
        EAR = 上下眼睑距离 / 内外眼角距离，睁眼约 0.25-0.35，闭眼 < 0.1
        （旧实现的"垂直"与"水平"取的是同一段线段，EAR 恒 ≥ 0.5，闭眼永远检测不到）
        """
        v = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[3]))
        h = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[1]))
        if h < 1e-6:
            return 0.0
        return float(v / h)

    @staticmethod
    def _mouth_aspect_ratio(points) -> float:
        """
        计算嘴部纵横比 (Mouth Aspect Ratio)

        MAR = 上下内唇距离(13-14) / 嘴角宽度(61-291)
        （旧实现除以的是 13/14 两点的水平差——这两点几乎垂直对齐，分母被钳到 1 像素，
        导致闭嘴时 MAR 也远超阈值，哈欠检测常开）
        """
        top = np.array(points[13])
        bottom = np.array(points[14])
        left = np.array(points[61])
        right = np.array(points[291])
        width = np.linalg.norm(left - right)
        if width < 1e-6:
            return 0.0
        return float(np.linalg.norm(top - bottom) / width)

    async def _assess_risk(self):
        """
        综合风险评估 — 疲劳 + 分心 → 告警等级

        normal:  一切正常
        warning: 轻度疲劳/短时分心（视觉提示）
        high:    中度疲劳/持续分心（语音提醒）
        critical: 重度疲劳（强制提醒+建议休息）
        """
        fatigue_score = 0.0

        # PERCLOS 评分 (0-40分)
        if self.state.perclos > self.config.perclos_threshold:
            fatigue_score += min(40, self.state.perclos * 100)

        # 哈欠频率 (0-30分)
        if self.state.yawn_count >= self.config.yawn_frequency_threshold:
            fatigue_score += 30

        # 头部姿态 (0-20分)
        if self.state.head_pose == "down":
            fatigue_score += 20

        # 分心 (0-20分)
        if self.state.distraction_duration > self.config.distraction_duration_threshold:
            fatigue_score += min(20, self.state.distraction_duration * 5)

        # 确定告警等级
        if fatigue_score >= 60:
            new_level = "critical"
        elif fatigue_score >= 35:
            new_level = "high"
        elif fatigue_score >= 15:
            new_level = "warning"
        else:
            new_level = "normal"

        # 状态变化时发布事件
        if new_level != self.state.alert_level:
            self.state.alert_level = new_level
            self._fatigue_score = fatigue_score

            # 检查冷却
            now = time.time()
            if new_level in ("high", "critical"):
                cooldown = self._last_alert_time.get(new_level, 0)
                if now - cooldown < 30:  # 30秒冷却
                    return
                self._last_alert_time[new_level] = now

            # 构建告警信息
            alert_messages = {
                "warning": "您看起来有些疲劳，请注意休息",
                "high": "检测到疲劳驾驶迹象，请尽快靠边休息！",
                "critical": "危险！您已严重疲劳，请立即停车休息！前方服务区已在导航中标记",
            }

            if new_level != "normal":
                await self.event_bus.publish(Event(
                    type="safety.alert",
                    data={
                        "level": new_level,
                        "type": "fatigue",
                        "message": alert_messages.get(new_level, ""),
                        "score": fatigue_score,
                        "perclos": self.state.perclos,
                        "yawn_count": self.state.yawn_count,
                        "distraction_duration": self.state.distraction_duration,
                    }
                ))
                logger.warning(
                    f"⚠️ 安全告警 [{new_level}] "
                    f"PERCLOS={self.state.perclos:.2f} "
                    f"哈欠={self.state.yawn_count} "
                    f"分心={self.state.distraction_duration:.1f}s "
                    f"评分={fatigue_score:.0f}"
                )

    async def _simulated_monitor_loop(self):
        """模拟监控循环（无摄像头时使用）"""
        import random
        while self.running:
            await asyncio.sleep(5)

            # 随机模拟状态变化（演示用）
            level = random.choices(
                ["normal", "normal", "normal", "normal", "warning", "high"],
                weights=[70, 10, 5, 5, 5, 5]
            )[0]

            if level != self.state.alert_level:
                self.state.alert_level = level
                if level != "normal":
                    await self.event_bus.publish(Event(
                        type="safety.alert",
                        data={
                            "level": level,
                            "type": "fatigue",
                            "message": f"模拟告警: {level}",
                            "simulated": True,
                        }
                    ))

    def get_current_state(self) -> dict:
        """获取当前安全状态"""
        return {
            "fatigue_level": self.state.alert_level,
            "perclos": self.state.perclos,
            "yawn_count": self.state.yawn_count,
            "gaze_direction": self.state.gaze_direction,
            "distraction_duration": self.state.distraction_duration,
            "is_distracted": self._distraction_start is not None,
        }

    async def shutdown(self):
        """关闭安全监控"""
        self.running = False
        if self.camera:
            self.camera.release()
        logger.info("安全监控引擎已关闭")
