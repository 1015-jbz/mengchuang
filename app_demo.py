"""
智能座舱多模态交互终端 — 统一 Web 界面
========================================
实时表情识别 + 语音对话，全部在一个页面内完成

技术方案:
- Flask 提供 MJPEG 实时视频流（稳定可靠）
- Gradio 提供对话 UI
- 两者嵌入同一页面，Flask 作为 Gradio 的 upstream 代理

运行方式:
  python app_demo.py
  浏览器打开 http://localhost:7860
"""
import sys
import io
import os
import time
import threading
import base64
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 加载 .env 中的环境变量（DeepSeek API Key）
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import numpy as np
import gradio as gr

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# OpenCV
# ============================================================
try:
    import cv2
    _CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    _smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    _lefteye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
    _righteye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")
    HAS_CV2 = True
    logger.info("OpenCV 多级联检测器就绪 (人脸+微笑+左眼+右眼)")
    # ONNX 深度学习表情识别模型
    _ort_session = None
    _ort_labels = None
    _ort_input_size = 260
    _model_path = Path(__file__).resolve().parent / "models" / "enet_b2_7.onnx"
    if _model_path.exists():
        import onnxruntime as _ort
        _ort_session = _ort.InferenceSession(str(_model_path), providers=['CPUExecutionProvider'])
        _ort_labels = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}
        logger.info(f"ONNX 表情识别模型已加载: enet_b2_7 (260x260, 7类)")
    else:
        logger.info("ONNX 模型未下载，使用启发式。FDM 下载后放到 models/enet_b2_7.onnx 即可")
except Exception as e:
    _face_cascade = None
    _smile_cascade = None
    HAS_CV2 = False
    logger.warning(f"OpenCV 不可用: {e}")

# MediaPipe 人脸关键点（468点，精准表情识别）
# MediaPipe 0.10.x 改用 Tasks API，需兼容新旧两种
_face_mesh = None
try:
    import mediapipe as mp
    # 尝试旧版 API (0.9.x)
    if hasattr(mp, 'solutions'):
        _mp_face_mesh = mp.solutions.face_mesh
        _face_mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        HAS_MEDIAPIPE = True
        logger.info("MediaPipe 人脸关键点就绪 (legacy API)")
    else:
        # 新版 Tasks API (0.10.x) — 需要模型文件，先跳过
        HAS_MEDIAPIPE = False
        _face_mesh = None
        logger.info("MediaPipe 0.10.x detected，使用增强启发式方案")
except ImportError:
    HAS_MEDIAPIPE = False
    logger.info("MediaPipe 未安装，使用增强启发式表情检测")

# ============================================================
# 共享状态（线程安全）
# ============================================================
_latest_frame = None          # 最新的带标注帧 (JPEG bytes)
_latest_emotion = "neutral"
_latest_confidence = 0.0
_lock = threading.Lock()

# TTS 异步队列：后台合成为避免阻塞 UI，完成后由定时器推送到前端
_pending_audio = None
_audio_lock = threading.Lock()

# ============================================================
# Flask MJPEG 视频流服务
# ============================================================
try:
    from flask import Flask, Response, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    logger.warning("Flask 未安装，视频流不可用")

flask_app = Flask(__name__)
_video_cap = None
_video_running = False

EMOTION_COLORS = {
    "happy": (0, 255, 128), "sad": (255, 128, 64), "angry": (0, 0, 255),
    "surprised": (0, 255, 255), "fearful": (128, 0, 128),
    "neutral": (180, 180, 180), "disgusted": (0, 128, 64),
}
EMOTION_ZH = {
    "happy": "开心", "sad": "悲伤", "angry": "愤怒",
    "surprised": "惊讶", "fearful": "恐惧", "neutral": "平静",
    "disgusted": "厌恶",
}

# ============================================================
# 语音角色选择 — edge-tts 中文语音库（均为微软神经 TTS）
# ============================================================
VOICE_OPTIONS = {
    # ── 动漫风 ──
    "xiaoyi":    "🎭 晓伊 · 活泼动漫少女 (Cartoon, Lively)",
    "yunxia":    "🎭 云夏 · 可爱动漫正太 (Cartoon, Cute)",
    # ── 温柔/日常 ──
    "xiaoxiao":  "🌸 晓晓 · 温柔姐姐 (Warm, 默认)",
    "yunxi":     "☀️ 云希 · 阳光少年 (Lively, Sunshine)",
    "yunjian":   "🔥 云健 · 热血青年 (Passion)",
    # ── 专业播报 ──
    "yunyang":   "📰 云扬 · 专业主播 (Professional)",
    # ── 方言趣味 ──
    "xiaobei":   "🤣 晓北 · 东北大碴子 (Dialect, Humorous)",
    "xiaoni":    "😄 晓妮 · 陕西嫽咋咧 (Dialect, Bright)",
}
VOICE_IDS = {k: f"zh-CN-{k.capitalize()}Neural" for k in VOICE_OPTIONS}
DEFAULT_VOICE = "xiaoyi"  # 默认改成动漫少女，更有趣

# 声线预设 — 一键切换风格
VOICE_STYLE_PRESETS = {
    "默认":      ("+0Hz",  "+0%"),
    "二次元萌音": ("+30Hz", "+15%"),   # 高音 + 稍快 ≈ 动漫少女
    "懒羊羊 🐑":  ("-20Hz", "-40%"),   # 低音 + 大幅放慢 ≈ 懒羊羊拖长音
    "热血少年":   ("+10Hz", "+25%"),   # 略高 + 快
    "沉稳大叔":   ("-25Hz", "-15%"),   # 低音 + 稍慢
    "可爱正太":   ("+40Hz", "+10%"),   # 很高 + 稍快
}

# 基线校准系统 — LBP 纹理特征基线
_baseline = None
_baseline_count = 0
_BASELINE_FRAMES = 60

def _onnx_predict_emotion(face_rgb):
    """深度学习表情识别 — ONNX EfficientNet-B2"""
    img = cv2.resize(face_rgb, (_ort_input_size, _ort_input_size)) / 255.0
    img[..., 0] = (img[..., 0] - 0.485) / 0.229
    img[..., 1] = (img[..., 1] - 0.456) / 0.224
    img[..., 2] = (img[..., 2] - 0.406) / 0.225
    x = img.transpose(2, 0, 1).astype('float32')[np.newaxis, ...]
    scores = _ort_session.run(None, {'input': x})[0][0]
    e_x = np.exp(scores - np.max(scores))
    probs = e_x / e_x.sum()
    pred = int(np.argmax(probs))
    return _ort_labels[pred], float(probs[pred])

def _extract_face_features(face_img_rgb):
    """提取面部特征向量（含级联检测器特征）"""
    gray = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    # 嘴部
    mouth_roi = gray[2*h//3:, w//6:5*w//6]
    _, mt = cv2.threshold(mouth_roi, 70, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = float(np.sum(mt == 255) / max(mt.size, 1))
    mouth_contrast = float(mouth_roi.std())
    left = gray[2*h//3:, :w//3]
    right = gray[2*h//3:, 2*w//3:]
    mouth_asym = abs(float(left.std()) - float(right.std()))
    # 眉心竖纹 (Sobel 垂直边缘)
    glabella = gray[h//8:3*h//8, w//3:2*w//3]
    sobel_x = cv2.Sobel(glabella, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(glabella, cv2.CV_64F, 0, 1, ksize=3)
    vertical = np.abs(sobel_x) - np.abs(sobel_y) * 0.5
    brow = float(np.sum(vertical > 40) / max(vertical.size, 1))
    # 眼睛大小（级联检测器检测到的眼睛面积）
    upper_face = gray[:h//2, :]
    le = _lefteye_cascade.detectMultiScale(upper_face, 1.1, 3, minSize=(15, 10))
    re = _righteye_cascade.detectMultiScale(upper_face, 1.1, 3, minSize=(15, 10))
    eye_area = 0.0
    if len(le) > 0 and len(re) > 0:
        # 取最大检测框面积
        le_area = max(e[2]*e[3] for e in le)
        re_area = max(e[2]*e[3] for e in re)
        eye_area = float((le_area + re_area) / 2) / (w * h)  # 归一化
    # 全局
    fmean = float(gray.mean())
    fstd = float(gray.std())
    return dark_ratio, mouth_asym, mouth_contrast, brow, fmean, fstd, eye_area

_last_onnx_result = ("neutral", 0.5)
_onnx_frame_counter = 0

def _sadness_boost(onnx_emotion, onnx_conf, face_img_rgb):
    """ONNX 后处理：AffectNet 数据集中悲伤样本偏少，模型对悲伤不够敏感。
    当 ONNX 预测为 neutral 但面部特征强烈指向悲伤时，提升为 sad。"""
    if onnx_emotion != "neutral":
        return onnx_emotion, onnx_conf

    # 提取面部特征做悲伤启发式判断
    gray = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    fmean = float(gray.mean())
    fstd = float(gray.std())

    # 嘴角区域：检测左右不对称 + 嘴角下垂特征
    mouth = gray[2*h//3:, w//6:5*w//6]
    left_mouth = gray[2*h//3:, :w//3]
    right_mouth = gray[2*h//3:, 2*w//3:]
    mouth_std = float(mouth.std())
    mouth_asym = abs(float(left_mouth.std()) - float(right_mouth.std()))

    # 眼睛区域面积
    upper = gray[:h//2, :]
    eye_area = 0.0
    if _lefteye_cascade is not None and _righteye_cascade is not None:
        le = _lefteye_cascade.detectMultiScale(upper, 1.1, 3, minSize=(15, 10))
        re = _righteye_cascade.detectMultiScale(upper, 1.1, 3, minSize=(15, 10))
        if len(le) > 0 and len(re) > 0:
            le_area = max(e[2]*e[3] for e in le)
            re_area = max(e[2]*e[3] for e in re)
            eye_area = float((le_area + re_area) / 2) / (w * h)

    # 悲伤特征: 面部偏暗 + 眼睛偏小 + 嘴部活动少 + 低对比度
    sad_score = 0
    if fmean < 120:      sad_score += 1   # 面部较暗
    if fstd < 30:         sad_score += 1   # 低对比度
    if eye_area < 0.04:   sad_score += 1   # 眼睛偏小（半闭眼）
    if mouth_std < 30:    sad_score += 1   # 嘴部不动
    if mouth_asym > 5:    sad_score += 1   # 嘴角不对称

    if sad_score >= 3:
        conf = min(0.75, 0.40 + sad_score * 0.08)
        return "sad", conf
    return onnx_emotion, onnx_conf


def detect_emotion_from_landmarks(face_img_rgb, frame_w, frame_h):
    # 深度学习模型优先（每5帧推理一次，其余用缓存）
    global _last_onnx_result, _onnx_frame_counter
    if _ort_session is not None:
        _onnx_frame_counter += 1
        if _onnx_frame_counter % 5 == 0:
            raw_emotion, raw_conf = _onnx_predict_emotion(face_img_rgb)
            # ONNX 后处理: 悲伤识别增强（AffectNet 悲伤样本偏少，模型易漏）
            _last_onnx_result = _sadness_boost(raw_emotion, raw_conf, face_img_rgb)
        return _last_onnx_result
    """
    基于个人基线的表情识别 —— 相对变化远优于绝对阈值。

    启动时自动采集平静脸建立基线，之后每帧与基线对比，
    检测特征偏离来判断表情。对不同人脸/光照自适应。
    """
    global _baseline, _baseline_count

    # 提取当前特征
    dark, asym, mcon, brow, fmean, fstd, eye_area = _extract_face_features(face_img_rgb)

    # 建立基线（前 N 帧平均）
    if _baseline_count < _BASELINE_FRAMES:
        if _baseline is None:
            _baseline = (dark, asym, mcon, brow, fmean, fstd, eye_area)
        else:
            w = _baseline_count / (_baseline_count + 1)
            _baseline = (
                _baseline[0] * w + dark * (1-w),
                _baseline[1] * w + asym * (1-w),
                _baseline[2] * w + mcon * (1-w),
                _baseline[3] * w + brow * (1-w),
                _baseline[4] * w + fmean * (1-w),
                _baseline[5] * w + fstd * (1-w),
                _baseline[6] * w + eye_area * (1-w),
            )
        _baseline_count += 1
        if _baseline_count == _BASELINE_FRAMES:
            logger.info(f"基线建立完成: dark={_baseline[0]:.4f} asym={_baseline[1]:.2f} "
                        f"mcon={_baseline[2]:.1f} brow={_baseline[3]:.4f} eye={_baseline[6]:.4f}")
        return "neutral", 0.5

    # 与基线对比，计算变化差值
    bdark, basym, bmcon, bbrow, bmean, bstd, beye = _baseline

    dark_delta = dark - bdark
    asym_delta = asym - basym
    brow_delta = brow - bbrow
    mcon_delta = mcon - bmcon
    fmean_delta = fmean - bmean
    eye_delta = eye_area - beye    # 眼睛面积变化 → 睁大/眯眼

    # === 多级联检测器融合（Haar + 像素特征）===
    h, w = face_img_rgb.shape[:2]
    eyes_wide = eye_delta > 0.002       # 眼睛明显睁大
    eyes_narrow = eye_delta < -0.001    # 眼睛明显变小
    mouth_open = dark_delta > 0.008     # 嘴明显张开(大)
    mouth_slight = dark_delta > 0.003   # 嘴微张
    brow_furrowed = brow_delta > 0.008  # 眉心明显竖纹
    face_dark = fmean_delta < -8        # 面部明显变暗

    # 😊 开心 — smile cascade（级联模型，最可靠）
    if _smile_cascade is not None:
        gray_lower = cv2.cvtColor(face_img_rgb[h//2:, :], cv2.COLOR_RGB2GRAY)
        for sf, mn in [(1.3, 15), (1.5, 10), (1.8, 8)]:
            smiles = _smile_cascade.detectMultiScale(
                gray_lower, sf, mn, minSize=(20, 15), maxSize=(w//2, h//3))
            if len(smiles) > 0:
                return "happy", 0.8

    # 😲 惊讶 — 嘴大张 + 眼睛睁大 + 眉心不皱(眉毛上扬)
    if mouth_open and eyes_wide and not brow_furrowed:
        return "surprised", min(0.9, 0.4 + dark_delta * 30 + eye_delta * 40)

    # 😨 恐惧 — 眼睛睁大 + 嘴微张 + 眉心可能皱（区别于惊讶）
    if eyes_wide and mouth_slight and not mouth_open:
        return "fearful", min(0.75, 0.3 + eye_delta * 50)

    # 😤 愤怒 — 眉心竖纹 + 眼睛不睁大 + 嘴不大张
    if brow_furrowed and not eyes_wide and not mouth_open:
        return "angry", min(0.8, 0.3 + brow_delta * 15)

    # 😢 悲伤 — 面部变暗 + 眼睛变小 + 嘴不张
    if face_dark and eyes_narrow and not mouth_open:
        return "sad", min(0.7, 0.3 + abs(fmean_delta) * 0.02)

    return "neutral", 0.5

    results = _face_mesh.process(face_img_rgb)
    if not results.multi_face_landmarks:
        return "neutral", 0.0

    lm = results.multi_face_landmarks[0]
    h, w = face_img_rgb.shape[:2]

    def pt(idx):
        return np.array([lm.landmark[idx].x * w, lm.landmark[idx].y * h])

    # --- 嘴部特征 ---
    lip_top = pt(13)      # 上唇中点
    lip_bottom = pt(14)   # 下唇中点
    lip_left = pt(61)     # 左嘴角
    lip_right = pt(291)   # 右嘴角

    mouth_open = np.linalg.norm(lip_top - lip_bottom)  # 张嘴程度
    mouth_width = np.linalg.norm(lip_left - lip_right)  # 嘴宽度
    mar = mouth_open / (mouth_width + 1e-6)             # 嘴部纵横比

    # 嘴角相对位置（判断上扬/下垂）
    lip_center_y = (lip_left[1] + lip_right[1]) / 2
    mouth_mid_y = (lip_top[1] + lip_bottom[1]) / 2
    corner_up = mouth_mid_y - lip_center_y  # 正=上扬(开心), 负=下垂(悲伤)

    # --- 眉毛特征 ---
    brow_left_in = pt(55)    # 左眉内侧
    brow_left_out = pt(46)   # 左眉外侧
    brow_right_in = pt(285)  # 右眉内侧
    brow_right_out = pt(276) # 右眉外侧
    eye_left_top = pt(159)   # 左眼上沿
    eye_right_top = pt(386)  # 右眼上沿

    brow_height = ((brow_left_in[1] + brow_right_in[1]) / 2 -
                   (eye_left_top[1] + eye_right_top[1]) / 2)
    brow_height_norm = brow_height / h  # 归一化眉毛高度

    # --- 眼睛特征 ---
    eye_left_bottom = pt(145)
    eye_right_bottom = pt(374)
    left_ear = (np.linalg.norm(pt(159) - pt(145)) /
                np.linalg.norm(pt(33) - pt(133)) + 1e-6)
    right_ear = (np.linalg.norm(pt(386) - pt(374)) /
                 np.linalg.norm(pt(362) - pt(263)) + 1e-6)
    avg_ear = (left_ear + right_ear) / 2  # 平均眼部纵横比

    # --- 分类逻辑 ---
    if mar > 0.55 and brow_height_norm > 0.03:
        return "surprised", min(0.9, mar * 1.2)
    if mar > 0.45:
        return "surprised", min(0.85, mar * 1.0)

    if corner_up > 3.0 and mar > 0.1:
        return "happy", min(0.85, 0.4 + corner_up * 0.05)

    if brow_height_norm < 0.005 and mar < 0.15 and corner_up < 0:
        return "angry", min(0.8, 0.5 - brow_height_norm * 10)

    if corner_up < -2.0 and mar < 0.2:
        return "sad", min(0.75, 0.4 + abs(corner_up) * 0.04)

    if avg_ear < 0.18:
        return "tired", 0.65

    if brow_height_norm > 0.04 and mar < 0.2:
        return "fearful", 0.55

    return "neutral", 0.6

def _capture_loop():
    """后台线程：持续捕获摄像头并检测表情"""
    global _video_cap, _video_running, _latest_frame, _latest_emotion, _latest_confidence

    # 尝试 DShow 后端
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        _video_running = False
        return

    logger.info("摄像头已连接，开始实时检测...")
    _video_cap = cap
    _video_running = True

    while _video_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        annotated = frame.copy()
        h, w = frame.shape[:2]

        emotion = "neutral"
        conf = 0.0
        face_detected = False

        # 优先用 MediaPipe 精准检测（468个关键点）
        if HAS_MEDIAPIPE and _face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = _face_mesh.process(rgb)
            if results.multi_face_landmarks:
                face_detected = True
                # 用 MediaPipe 检测表情
                emotion, conf = detect_emotion_from_landmarks(rgb, w, h)
                # 画人脸框（用关键点推算）
                lm = results.multi_face_landmarks[0]
                xs = [l.x * w for l in lm.landmark]
                ys = [l.y * h for l in lm.landmark]
                x, y, fw, fh = int(min(xs)), int(min(ys)), int(max(xs)-min(xs)), int(max(ys)-min(ys))

                color = EMOTION_COLORS.get(emotion, (180, 180, 180))
                label = f"{EMOTION_ZH.get(emotion, emotion)} ({conf:.0%})"
                # 画框（加粗）
                cv2.rectangle(annotated, (x, y), (x+fw, y+fh), color, 3)
                # 画标签 — 大号粗体醒目
                font = cv2.FONT_HERSHEY_DUPLEX
                scale = 1.2
                thick = 3
                (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
                # 标签背景
                cv2.rectangle(annotated, (x-3, y-th-20), (x+tw+15, y+5), color, -1)
                # 标签文字（黑色在亮色背景上，白色在暗色背景上）
                text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
                cv2.putText(annotated, label, (x+5, y-5),
                            font, scale, text_color, thick)
        else:
            # 降级：OpenCV Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(64, 64))
            if len(faces) > 0:
                (x, y, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
                face_img = frame[y:y+fh, x:x+fw]
                emotion, conf = detect_emotion_from_landmarks(
                    cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), fw, fh)
                face_detected = True

                color = EMOTION_COLORS.get(emotion, (180, 180, 180))
                label = f"{EMOTION_ZH.get(emotion, emotion)} ({conf:.0%})"
                font = cv2.FONT_HERSHEY_DUPLEX
                scale = 1.2; thick = 3
                cv2.rectangle(annotated, (x, y), (x+fw, y+fh), color, 3)
                (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
                cv2.rectangle(annotated, (x-3, y-th-20), (x+tw+15, y+5), color, -1)
                text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
                cv2.putText(annotated, label, (x+5, y-5),
                            font, scale, text_color, thick)

        # 编码为 JPEG
        _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])

        with _lock:
            _latest_frame = jpeg.tobytes()
            _latest_emotion = emotion
            _latest_confidence = conf

    cap.release()

@flask_app.route('/video_feed')
def video_feed():
    """MJPEG 视频流端点"""
    def generate():
        while True:
            with _lock:
                frame = _latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)  # ~25 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@flask_app.route('/emotion_status')
def emotion_status():
    """表情状态 JSON 端点"""
    with _lock:
        return {"emotion": _latest_emotion, "confidence": _latest_confidence,
                "label": EMOTION_ZH.get(_latest_emotion, "未知")}

# ============================================================
# 语音识别（可选）
# ============================================================
asr_model = None
try:
    from faster_whisper import WhisperModel
    import os as _os
    # 优先查 D 盘自定义路径（企业网 SSL 拦截导致正常下载失败，模型手动下载到 D 盘）
    _whisper_path = "D:/huggingface_cache/models--Systran--faster-whisper-tiny/snapshots/tiny"
    _cache_path = _os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-whisper-tiny")
    if _os.path.isdir(_whisper_path):
        asr_model = WhisperModel(_whisper_path, device="cpu", compute_type="int8", local_files_only=True)
        logger.info("faster-whisper 就绪 (D盘)")
    elif _os.path.isdir(_cache_path):
        asr_model = WhisperModel("tiny", device="cpu", compute_type="int8", local_files_only=True)
        logger.info("faster-whisper 就绪 (C盘缓存)")
    else:
        logger.info("Whisper 模型未缓存，语音识别使用文本降级")
except Exception:
    logger.info("语音识别使用文本降级模式")

def transcribe(audio_path):
    if audio_path is None or asr_model is None:
        return ""
    try:
        segments, _ = asr_model.transcribe(audio_path, language="zh", beam_size=5)
        return " ".join(s.text.strip() for s in segments)
    except Exception:
        return ""

# ============================================================
# 对话引擎
# ============================================================
# ═══════════════════════════════════════════════════════════
# 智能座舱对话引擎 "小航" — DeepSeek 大模型优先 + 模板兜底
# ═══════════════════════════════════════════════════════════

import re as _re, random as _random, datetime as _dt

# ── DeepSeek 大模型客户端 ──
_deepseek_client = None

def _get_deepseek():
    """获取 DeepSeek 客户端（启动时预热，首次调用不额外耗时）"""
    global _deepseek_client
    if _deepseek_client is None:
        try:
            import os as _os
            from openai import OpenAI
            _key = _os.getenv("DEEPSEEK_API_KEY")
            if _key:
                _deepseek_client = OpenAI(
                    api_key=_key, base_url="https://api.deepseek.com",
                    timeout=10.0, max_retries=1,
                )
                # 预热连接：发一个极短请求，让后续调用复用 HTTP 连接
                try:
                    _deepseek_client.chat.completions.create(
                        model="deepseek-chat", messages=[{"role":"user","content":"hi"}],
                        max_tokens=5, temperature=0,
                    )
                except: pass
                logger.info("DeepSeek 大模型已连接并预热")
            else:
                _deepseek_client = False
                logger.warning("未设置 DEEPSEEK_API_KEY，使用本地模板回复")
        except Exception as e:
            _deepseek_client = False
            logger.warning(f"DeepSeek 初始化失败: {e}，使用本地模板回复")
    return _deepseek_client if _deepseek_client is not False else None


# ── 大模型对话记忆 ──
_llm_history: list = []  # [{role, content}, ...]
_LLM_SYSTEM_PROMPT = """你是"小航"，一个运行在智能汽车座舱里的 AI 助手。你的性格温暖、贴心、靠谱。

核心设定：
- 你运行在汽车本地芯片上，所有对话数据不会上传云端
- 你能控制空调、车窗、座椅、灯光、导航、音乐等车载设备
- 你时刻关注驾驶安全，察觉疲劳/分心时会主动提醒
- 你能通过车内摄像头感知驾驶员的情绪，并做出相应关怀

回复规则：
1. 简短温暖——用户可能在开车，回复控制在一两句话内（除非用户要求详细说明）
2. 安全第一——任何可能分散注意力的操作都提醒"安全驾驶"
3. 情绪感知——察觉用户情绪低落时主动安慰，开心时一起开心
4. 口语化自然——不要像机器人，像坐在副驾的朋友
5. 如果用户说的跟驾驶/车辆无关，就用日常朋友聊天的语气回应
6. 不要用 markdown 格式，纯文字回复"""


def _llm_chat(user_text: str, emo_hint: str = "") -> str | None:
    """调用 DeepSeek 大模型生成回复，失败则返回 None 触发模板兜底"""
    client = _get_deepseek()
    if client is None:
        return None
    try:
        _llm_history.append({"role": "user", "content": user_text})
        ctx = _llm_history[-12:] if len(_llm_history) > 12 else _llm_history
        messages = [{"role": "system", "content": _LLM_SYSTEM_PROMPT}]
        if emo_hint:
            messages.append({"role": "system", "content": emo_hint})
        messages += ctx
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=120,
            temperature=0.6,
        )
        reply = resp.choices[0].message.content.strip()
        _llm_history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.warning(f"DeepSeek 调用失败: {e}，回退模板回复")
        if _llm_history and _llm_history[-1]["role"] == "user":
            _llm_history.pop()
        return None

# ── 对话上下文（简单短期记忆）──
_dialog_memory = {"last_topic": None, "greeted": False, "turn": 0}

# ── 意图路由：关键词 → (领域, 意图) ──
_ROUTES = [
    # 车辆控制
    (["空调", "温度", "冷气", "暖气", "通风", "除雾", "冷", "热"], "车辆控制", "空调"),
    (["车窗", "窗户", "天窗", "开窗", "关窗", "透气"], "车辆控制", "车窗"),
    (["座椅", "座位", "加热", "按摩", "通风座椅"], "车辆控制", "座椅"),
    (["灯光", "大灯", "远光", "近光", "氛围灯", "阅读灯", "双闪"], "车辆控制", "灯光"),
    (["驾驶模式", "运动模式", "经济模式", "舒适模式", "雪地模式"], "车辆控制", "驾驶模式"),
    # 导航出行
    (["导航", "去", "怎么走", "路线", "开到", "前往", "带我去"], "导航出行", "目的地"),
    (["还有多远", "多久到", "还要多久", "什么时候到"], "导航出行", "路况"),
    (["堵车", "路况", "拥堵", "前方", "绕路", "避开"], "导航出行", "路况"),
    # 影音娱乐
    (["播放", "放歌", "音乐", "来一首", "听歌", "我想听", "放一首"], "影音娱乐", "音乐"),
    (["音量", "大声", "小声", "静音", "吵", "轻一点"], "影音娱乐", "音量"),
    (["广播", "电台", "FM", "收音机", "调频"], "影音娱乐", "电台"),
    (["笑话", "讲个故事", "有趣的事", "段子", "开心一下"], "影音娱乐", "娱乐"),
    # 安全守护
    (["胎压", "油量", "电量", "水温", "里程", "续航", "还剩多少"], "安全守护", "状态"),
    (["救命", "报警", "求助", "SOS", "紧急", "危险", "出事了"], "安全守护", "紧急"),
    (["疲劳", "开了多久", "驾驶时间", "连续驾驶"], "安全守护", "驾驶时长"),
    # 情感陪伴
    (["心情", "难过", "不开心", "郁闷", "烦", "累", "困", "伤心",
      "压力", "焦虑", "紧张", "孤独", "无聊", "emo"], "情感陪伴", "倾诉"),
    (["你好", "嗨", "早上好", "晚上好", "下午好", "小航", "在吗"], "情感陪伴", "问候"),
    (["谢谢", "感谢", "辛苦了", "你真棒", "厉害"], "情感陪伴", "夸奖"),
    (["再见", "拜拜", "晚安", "我走了", "关机"], "情感陪伴", "道别"),
    # 信息查询
    (["天气", "下雨", "几度", "晴", "阴", "温度", "刮风", "雾霾"], "信息查询", "天气"),
    (["时间", "几点", "日期", "今天几号", "星期几"], "信息查询", "时间"),
    (["附近", "周边", "最近的", "找个", "加油站", "充电桩", "停车场", "服务区", "餐厅"], "信息查询", "周边"),
    (["你叫什么", "你是谁", "你的功能", "你能做什么", "介绍一下"], "信息查询", "自我介绍"),
]


# ── 领域回复库：每条意图有多组自然回答，随机抽取避免机械感 ──
_REPLIES = {
    "车辆控制/空调": [
        lambda t: f"好的，空调已设为{_re.search(r'(\d+)\s*度', t).group(1) if _re.search(r'(\d+)\s*度', t) else '舒适'}°C，车内很快就能到你想要的温度~",
        lambda t: f"收到，空调温度已调节。当前车内{_re.search(r'(\d+)\s*度', t).group(1) if _re.search(r'(\d+)\s*度', t) else '22'}°C，需要再调整随时跟我说。",
        lambda t: "空调已经打开了，保持这个温度会很舒服的。",
    ],
    "车辆控制/车窗": [
        lambda t: "好的，车窗已为您操作。如果觉得风太大了随时告诉我关小一点。",
        lambda t: "收到，车窗已调节。开车时注意安全，别让风直接吹到眼睛哦。",
        lambda t: "天窗已打开，享受一下自然风吧~不过高速行驶时建议关闭天窗降低风噪。",
    ],
    "车辆控制/座椅": [
        lambda t: "座椅加热已开启，冬天暖暖的~大概两分钟就能感受到温度了。",
        lambda t: "座椅已调节好。长途驾驶的话，建议每隔一段时间微调一下坐姿，对腰椎好。",
        lambda t: "座椅按摩功能已启动。工作了一天累了，好好享受一下吧！",
    ],
    "车辆控制/灯光": [
        lambda t: "灯光已切换。夜间行车建议使用近光灯，会车时记得切换哦，安全第一！",
        lambda t: "氛围灯已打开。开车时调暗一点可以减少视觉疲劳，你试试看~",
        lambda t: "好的，灯光已调节。最近天黑得早，记得及时开灯。",
    ],
    "车辆控制/驾驶模式": [
        lambda t: "驾驶模式已切换。不同模式会影响油门响应和悬挂硬度，你可以感受一下变化。",
        lambda t: "好的，模式已更改。运动模式动力响应更快但油耗会稍高，经济模式更省油~",
    ],
    "导航出行/目的地": [
        lambda t: f"正在规划前往{_re.sub(r'(导航到?|去|怎么到|开到|前往|带我去)\s*', '', t).strip() or '目的地'}的路线，预计需要约30分钟，路况看起来还不错。",
        lambda t: f"好的，导航已设置。我帮你看了下，目前这条路上暂时没有拥堵，可以放心出发~",
        lambda t: f"路线规划好了！沿途会经过3个服务区，需要休息时随时叫我。一路顺风！",
    ],
    "导航出行/路况": [
        lambda t: "前方路况正常，没有拥堵～按照现在的速度，大概还需要半小时。",
        lambda t: "我看了下实时路况，目前这段路比较顺畅。不过前面两公里处有个红绿灯多的地方，稍微慢一点。",
        lambda t: "为你找到一条更快的小路，能省大约10分钟，要换路线吗？不过小路比较窄，开慢一点。",
    ],
    "影音娱乐/音乐": [
        lambda t: f"好的，正在为你播放歌曲~顺手帮你调到了最合适的音量，享受音乐的同时也要注意安全驾驶哦！",
        lambda t: f"这首歌我也喜欢！已经加到播放列表了。想换歌或者调音量随时叫我。",
        lambda t: f"播放器已就绪，接下来请欣赏~如果你跟着哼两句就更好啦（不过别忘了看路）！",
    ],
    "影音娱乐/音量": [
        lambda t: "音量已调好。太高的话会听不到外界声音，安全起见我帮你控制在一个合适的范围。",
        lambda t: "好的，音量已调节。开车听音乐是好的，但也要能听到喇叭声哦~",
    ],
    "影音娱乐/电台": [
        lambda t: "正在连接FM广播... 信号好的话马上就有声音了。现在这个时段应该有不少好节目。",
        lambda t: "广播已打开。你可以告诉我喜欢听什么类型的——新闻、音乐、还是交通路况？",
    ],
    "影音娱乐/娱乐": [
        lambda t: _random.choice([
            "有一天，0和8在街上遇到。0不屑地说：\"胖成这样还系腰带！\"8说：\"你也好不到哪去，胖成个球了。\"",
            "程序员去面试，面试官问：\"你毕业到现在才两年，怎么写有五年工作经验？\"程序员答：\"加班加的。\"",
            "为什么程序员总是搞混圣诞节和万圣节？因为Oct 31等于Dec 25！",
            "面包走在路上，突然饿了，就把自己吃了。",
            "程序员最讨厌康熙的哪个儿子？胤禩，因为他是八阿哥(bug)。",
        ]),
        lambda t: "好，给你讲个小故事：从前有一个爱发明的人，他造了一辆车。这辆车最特别的地方是——它必须车主笑着才能启动。所以每天上班前，车主都会对着后视镜里的自己笑一笑。科学证明，假笑也能骗过大脑分泌多巴胺哦！",
    ],
    "安全守护/状态": [
        lambda t: "车辆各系统运行正常，可以放心驾驶。😊 胎压2.3-2.5 bar、油量充足、电量健康——一切OK！",
        lambda t: "我刚检查了一遍，各项指标都在安全范围内。现在的续航大约还能跑400公里，够你今天的行程了。",
        lambda t: "系统自检通过，没有异常。不过还是要提醒你，长途驾驶前最好自己再下车看一眼轮胎~",
    ],
    "安全守护/紧急": [
        lambda t: "已触发紧急求助！正在联系道路救援和紧急联系人，请保持冷静。我已经把你的位置发送出去了，救援大约15分钟内到达。",
        lambda t: "求救信号已发出！同时已自动打开双闪灯。不要慌，安全靠边停车，等待救援。我一直在线。",
    ],
    "安全守护/驾驶时长": [
        lambda t: "你已经连续驾驶快两个小时了，建议在前方服务区休息一下。安全比准时重要，歇15分钟喝口水再走。",
        lambda t: "根据安全规范，连续驾驶2小时就应该休息。你现在的注意力已经开始下降了，前方3公里有个服务区，去那里歇歇吧？",
    ],
    "情感陪伴/问候": [
        lambda t: _random.choice([
            "你好呀！我是小航，你的智能座舱伙伴。不管是导航、放歌还是陪你聊天，我都在！今天有什么计划吗？",
            "嗨！今天气色不错~我是小航，随时准备为你服务。要出发了吗？",
            "晚上好！辛苦了，终于可以放松一下了。我帮你把氛围灯调暗，放首舒缓的音乐怎么样？",
            "早上好！新的一天开始了~咖啡喝了吗？没喝也别在车里喝，不安全哈哈。今天去哪里呀？",
        ]),
        lambda t: "我在呢！有什么需要随时叫我。导航、音乐、空调、陪你聊天——这些我都会~",
    ],
    "情感陪伴/倾诉": [
        lambda t: _random.choice([
            "我懂那种感觉……生活有时候确实不容易。不过能说出来就好，我一直在这里听着。",
            "人都会有情绪低落的时候，这不代表你脆弱，恰恰说明你是个有血有肉的人。想聊就聊吧，不想聊我也可以安静地陪着你。",
            "有时候最好的疗愈就是有人陪着。虽然我是个AI，但我真的希望你能开心起来。要我放一首轻松的歌吗？",
            "你知道吗，心理学上说，把情绪说出来就已经解决了一半。你已经迈出了最重要的一步。剩下的，我们慢慢来。",
            "在这个小小的座舱里，你是最安全、最被接纳的。一切都会好起来的，我相信。",
        ]),
        lambda t: "你困了的话我们就在前面停一下？疲劳驾驶是安全最大的敌人，打个盹15分钟就能恢复不少精力。",
        lambda t: "累了就歇歇。我帮你盯着周围，窗户开一点让新鲜空气进来？",
    ],
    "情感陪伴/夸奖": [
        lambda t: _random.choice([
            "谢谢！能得到你的认可我真的很开心~我会继续努力的！",
            "哈哈，被你夸得都要飘起来了！不过认真说，我的每一点进步都离不开你的使用和反馈。",
        ]),
        lambda t: "不客气，这是我应该做的！你也是个很好的人，跟你相处很愉快~",
    ],
    "情感陪伴/道别": [
        lambda t: _random.choice([
            "再见！路上注意安全，到了记得给我报个平安~（虽然我收不到哈哈）",
            "晚安！好好休息，明天见~记得锁车哦！",
            "拜拜！期待下次见面。我会一直在这里等你的。",
        ]),
    ],
    "信息查询/天气": [
        lambda t: "今天天气整体不错，多云转晴，气温22~28°C，适合出行。不过傍晚可能有点小雨，建议带把伞以防万一。",
        lambda t: "我看了下天气，未来几小时都是晴天，温度舒适。适合开车兜风！不过紫外线比较强，记得戴墨镜。",
        lambda t: "这几天降温比较明显，建议出发前热车几分钟，开启座椅加热。路上注意路面可能结冰，保持车距。",
    ],
    "信息查询/时间": [
        lambda t: f"现在是{_dt.datetime.now().strftime('%H:%M')}，{_dt.datetime.now().strftime('%Y年%m月%d日')}，星期{['一','二','三','四','五','六','日'][_dt.datetime.now().weekday()]}。还有什么需要吗？",
    ],
    "信息查询/周边": [
        lambda t: "正在为你搜索附近...最近的加油站在前方2.3公里，服务区在5公里左右，还有一家评分不错的川菜馆在3公里处。需要导航到哪个？",
        lambda t: "附近有挺多选择的：左边800米有家便利店，右边2公里有个大型商场，前面5公里是高速服务区。想去哪里我帮你导航？",
    ],
    "信息查询/自我介绍": [
        lambda t: "我叫小航，是你的智能座舱伙伴！我运行在端侧芯片上，所有数据都在本地处理，不会上传云端哦。我可以帮你导航、调节车内设备、播放音乐、查天气、陪你聊天，还会一直关注你的驾驶状态保障安全。简单说——开车时你需要的一切，尽量帮你搞定！",
        lambda t: "我是小航~ 一个运行在车机本地芯片上的AI助手。不联网也能工作，所以你的隐私绝对安全。我能做的主要是：车辆控制（空调车窗灯光等）、导航、娱乐、安全监控、情感陪伴。有什么需要的尽管问我！",
    ],
}


def generate_reply(text, emotion="neutral"):
    """根据用户输入 + 情绪状态生成自然回复 — 优先 DeepSeek 大模型"""
    t = text.strip()

    # ── 优先: DeepSeek 大模型（真正理解语义，多轮记忆）──
    if t:
        # 情绪作为 system 级提示注入，不污染用户原话
        emo_hint = {
            "sad": "[系统感知：驾驶员表情显示悲伤，请在回复中给予温暖安慰]",
            "angry": "[系统感知：驾驶员表情显示愤怒，请帮助他冷静下来，把注意力放在安全驾驶上]",
            "fearful": "[系统感知：驾驶员表情显示恐惧/紧张，请安抚情绪，强调安全]",
            "tired": "[系统感知：驾驶员表情显示疲惫，请建议休息，提醒疲劳驾驶风险]",
            "happy": "[系统感知：驾驶员心情不错，可以开朗活泼地回应]",
        }.get(emotion, "")
        llm_reply = _llm_chat(t, emo_hint)
        if llm_reply:
            return llm_reply

    # ── 离线兜底: 模板引擎（无网络/API不可用时自动切换）──
    _dialog_memory["turn"] += 1

    # ── 意图路由 ──
    matched_domain = None
    matched_intent = None
    for keywords, domain, intent in _ROUTES:
        if any(kw in t for kw in keywords):
            matched_domain = domain
            matched_intent = intent
            break

    if matched_domain:
        _dialog_memory["last_topic"] = (matched_domain, matched_intent)
        key = f"{matched_domain}/{matched_intent}"
        pool = _REPLIES.get(key)
        if pool:
            return _random.choice(pool)(t)
        return "好的，收到。我正在处理你的请求~"

    # ── 空输入 ──
    if not t:
        return "我在听呢，有什么想说的吗？"

    # ── 短输入（可能是口头语/感叹）──
    if len(t) <= 2:
        shorts = ["嗯嗯，在呢。", "好的~", "收到！", "我听着呢。", "继续说呀，我在。"]
        return _random.choice(shorts)

    # ── 默认闲聊：结合情绪 ──
    emo_replies = {
        "sad":     ["虽然不知道具体发生了什么，但我想告诉你——你不是一个人。放首歌给你听好吗？",
                     "听你说话感觉你今天心情不太好。没关系，在这个座舱里，你可以完全放松。"],
        "angry":   ["我能感觉到你有些烦躁。深呼吸，把注意力放在前方的路上，其他的都先放一边。",
                     "开车时不开心是很危险的。要不要我讲个笑话帮你转换一下心情？"],
        "fearful": ["别担心，我一直在帮你看着路况。慢慢来，安全比速度重要。",
                     "紧张的时候可以先靠边停一下，调整呼吸。没什么比你的安全更重要。"],
        "happy":   ["看你心情不错，我也跟着开心！今天一定是很棒的一天~",
                     "真好！保持好心情，开车也会更安全。有什么好事分享一下吗？"],
        "tired":   ["你听起来需要休息。前方最近的能停车的地方大概2公里，要导航过去吗？",
                     "安全提醒：疲惫驾驶和酒驾一样危险。哪怕休息10分钟也会好很多。"],
    }

    if emotion in emo_replies:
        return _random.choice(emo_replies[emotion])

    # ── 通用闲聊：温暖但有变化 ──
    general = [
        "好的，收到。作为你的座舱伙伴，我会全力保障你的安全和舒适。",
        "明白。有什么需要随时告诉我——不管是调空调、导航还是陪你聊天~",
        "嗯嗯，我记下了。你开车时尽管吩咐我，不用分心去操作那些按钮。",
        "好的。对了，你开车时如果觉得无聊，可以让我讲笑话、放音乐、或者陪你聊聊天。",
        "收到。顺便提醒一下，你今天的驾驶评分目前是A级，继续保持哦！",
        "我一直在。无论是路上的事还是心里的事，都可以跟我说。",
    ]
    return _random.choice(general)

# ═══════════════════════════════════════════════════════════
# TTS 语音合成 — edge-tts 优先（8 种中文语音可选）→ pyttsx3 兜底
# ═══════════════════════════════════════════════════════════

# ── 后台 TTS 结果推送 ──
def _update_pending_audio(path):
    """后台线程回调：TTS 完成后将路径写入共享变量"""
    global _pending_audio
    if path:
        with _audio_lock:
            _pending_audio = path


def get_pending_audio():
    """定时器回调：有新音频时推送到前端"""
    global _pending_audio
    with _audio_lock:
        path = _pending_audio
        _pending_audio = None
    return path if path else gr.skip()


def tts_speak(text, voice_key=None, pitch="+0Hz", rate="+0%"):
    """合成语音文件并返回路径。edge-tts 2s 超时，超时/失败则 pyttsx3 兜底"""
    if not text:
        return None
    import tempfile
    stamp = int(time.time() * 1000)
    voice_id = VOICE_IDS.get(voice_key or DEFAULT_VOICE, VOICE_IDS[DEFAULT_VOICE])
    use_ssml = (pitch != "+0Hz" or rate != "+0%")
    tts_text = _build_ssml(text, voice_id, pitch, rate) if use_ssml else text

    # ── 优先: edge-tts（8 种中文语音，含动漫/方言/男声/女声）──
    out_mp3 = Path(tempfile.gettempdir()) / f"cockpit_tts_{stamp}.mp3"
    mp3_ready = [False]

    def _edge():
        try:
            import edge_tts, asyncio
            async def _gen():
                comm = edge_tts.Communicate(tts_text, voice_id)
                await comm.save(str(out_mp3))
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_gen())
            finally:
                loop.close()
            if out_mp3.exists() and out_mp3.stat().st_size > 0:
                mp3_ready[0] = True
        except Exception as e:
            logger.warning(f"edge-tts 语音合成失败: {e}")

    t = threading.Thread(target=_edge, daemon=True)
    t.start()
    t.join(timeout=4.0)  # 网络慢时 edge-tts 可能需 3-4 秒

    if mp3_ready[0]:
        return str(out_mp3)

    # ── 兜底: pyttsx3 离线（edge-tts 超时或失败时用）──
    try:
        out_wav = Path(tempfile.gettempdir()) / f"cockpit_tts_{stamp}.wav"
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        # 尝试根据语音角色匹配系统中文语音
        male_keys = {"yunxia", "yunxi", "yunjian", "yunyang"}
        if voice_key in male_keys:
            target = next((v for v in voices if v.id != voices[0].id), None)
        else:
            target = voices[0] if voices else None
        if target:
            engine.setProperty('voice', target.id)
        engine.save_to_file(text, str(out_wav))
        engine.runAndWait()
        engine.stop()
        if out_wav.exists() and out_wav.stat().st_size > 0:
            return str(out_wav)
    except Exception:
        pass

    return None


def _build_ssml(text, voice_id, pitch="+0Hz", rate="+0%"):
    """构建带声线调节的 SSML"""
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="zh-CN">'
        f'<voice name="{voice_id}">'
        f'<prosody pitch="{pitch}" rate="{rate}">'
        f'{text}'
        f'</prosody>'
        f'</voice>'
        f'</speak>'
    )



# ============================================================
# Gradio UI 回调
# ============================================================

def get_emotion_status():
    """获取当前表情状态"""
    with _lock:
        emo = _latest_emotion
        conf = _latest_confidence
    label = EMOTION_ZH.get(emo, "未知")
    emoji = {"happy": "# 😊 开心", "sad": "# 😢 悲伤", "angry": "# 😤 愤怒",
             "surprised": "# 😲 惊讶", "fearful": "# 😨 恐惧",
             "neutral": "# 😐 平静"}.get(emo, "# 😐 等待中")
    return f"{emoji}", conf, {
        "表情": label,
        "置信度": f"{conf:.0%}",
        "人脸检测": "已检测" if conf > 0 else "未检测或等待中",
    }

def process_text(text, chat_hist, voice_key, pitch_val, rate_val):
    """处理文本输入 — pitch_val/rate_val 直接来自滑块，支持 DIY 任意调节"""
    if chat_hist is None:
        chat_hist = []
    if not text or not text.strip():
        return chat_hist, "请输入内容", None

    text = text.strip()

    with _lock:
        emo = _latest_emotion

    try:
        reply = generate_reply(text, emo)
    except Exception as e:
        reply = f"抱歉，出错了: {e}"

    # TTS 后台合成，不阻塞文本回复（文字秒回，语音稍后自动播放）
    threading.Thread(
        target=lambda: _update_pending_audio(
            tts_speak(reply, voice_key, f"{pitch_val:+d}Hz", f"{rate_val:+d}%")),
        daemon=True
    ).start()

    # Gradio 6.0 使用新格式
    chat_hist.append({"role": "user", "content": text})
    chat_hist.append({"role": "assistant", "content": reply})

    # 找意图
    intent = "闲聊"
    for keywords, d, i in _ROUTES:
        if any(k in text for k in keywords):
            intent = f"{d}/{i}"
            break

    status = f"意图: {intent} | 情绪: {EMOTION_ZH.get(emo, '?')}"
    return chat_hist, status, None  # 音频由后台线程合成后经定时器推送到前端

def process_audio(audio_path, chat_hist, voice_key, pitch_val, rate_val):
    """处理语音输入"""
    chat_hist = chat_hist or []
    if audio_path is None:
        return chat_hist, "", None
    text = transcribe(audio_path)
    if not text:
        return chat_hist, "(未识别到语音)", None
    return process_text(text, chat_hist, voice_key, pitch_val, rate_val)

# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    # 修复 Windows GBK
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("[Smart Cockpit] 智能座舱多模态交互终端")
    print("=" * 60)
    print(f"  表情识别: {'MediaPipe' if HAS_MEDIAPIPE else '增强启发式'}")
    print(f"  语音识别: {'faster-whisper' if asr_model else '文本降级模式'}")
    print(f"  TTS:      edge-tts")
    print(f"  视频流:   Flask MJPEG (内嵌)")
    print("-" * 60)
    print("  打开浏览器: http://localhost:7860")
    print("=" * 60)

    # 预热 DeepSeek 连接（让首次 API 调用复用已建立的 HTTPS 连接）
    threading.Thread(target=_get_deepseek, daemon=True).start()

    # 启动摄像头采集线程
    if HAS_CV2 and _face_cascade is not None:
        t = threading.Thread(target=_capture_loop, daemon=True)
        t.start()
        time.sleep(1)  # 等待摄像头初始化

    # 启动 Flask 视频流（在独立线程）
    if HAS_FLASK:
        flask_thread = threading.Thread(
            target=lambda: flask_app.run(host='0.0.0.0', port=7861,
                                         debug=False, use_reloader=False),
            daemon=True
        )
        flask_thread.start()
        logger.info("Flask 视频流: http://localhost:7861/video_feed")
    else:
        logger.warning("Flask 未安装，无视频流")

    # Gradio UI
    with gr.Blocks(title="Smart Cockpit") as demo:
        gr.Markdown("# 智能座舱多模态交互终端")
        gr.Markdown("### 基于 LoongArch 端侧 AI | 实时表情识别 | 智能语音对话")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 实时表情识别")

                if HAS_FLASK:
                    # 直接嵌入 MJPEG 流。src 用页面自身的 hostname 拼接，
                    # 避免硬编码 localhost 导致手机/投屏等外部设备访问时视频黑屏
                    gr.HTML("""
                    <div style="border:2px solid #00d4aa; border-radius:10px; overflow:hidden; background:#000;">
                        <img src="invalid://bootstrap" style="width:100%; display:block;"
                             onerror="if(!this.dataset.boot){this.dataset.boot=1;this.src=location.protocol+'//'+location.hostname+':7861/video_feed';}else{this.onerror=null;this.parentElement.innerHTML='<p style=color:red;padding:20px;>摄像头未就绪，请确认摄像头已连接</p>';}">
                    </div>
                    <p style="text-align:center;color:#888;font-size:12px;">MJPEG 实时流 · 25 FPS</p>
                    """)
                else:
                    gr.Markdown("*视频流不可用 (Flask 未安装)*")

                emotion_display = gr.Markdown("# 😐 等待中...")
                emotion_bar = gr.Slider(0, 1, value=0, label="情绪置信度", interactive=False)
                status_box = gr.JSON(
                    value={"表情": "等待", "置信度": "0%", "人脸检测": "等待"},
                    label="检测状态",
                )

            with gr.Column(scale=2):
                gr.Markdown("### 智能语音对话")
                chatbot = gr.Chatbot(label="对话记录", height=400)

                with gr.Row():
                    mic = gr.Audio(sources=["microphone"], type="filepath", label="🎤 语音输入")
                    txt = gr.Textbox(placeholder="或在这里打字...", label="文字输入", scale=2)

                with gr.Row():
                    voice_selector = gr.Dropdown(
                        choices=[(label, key) for key, label in VOICE_OPTIONS.items()],
                        value=DEFAULT_VOICE,
                        label="🎙️ 语音角色",
                        interactive=True,
                        scale=1,
                    )
                    voice_style = gr.Dropdown(
                        choices=list(VOICE_STYLE_PRESETS.keys()),
                        value="默认",
                        label="🎚️ 风格预设",
                        interactive=True,
                        scale=1,
                    )

                with gr.Row():
                    pitch_slider = gr.Slider(
                        -50, 50, value=0, step=5,
                        label="🎵 音高偏移 (Hz) — 负数=低沉懒羊羊，正数=尖细萌音",
                        interactive=True,
                    )
                    rate_slider = gr.Slider(
                        -50, 50, value=0, step=5,
                        label="⏩ 语速偏移 (%) — 负数=拖长音，正数=快语速",
                        interactive=True,
                    )

                with gr.Row():
                    send = gr.Button("发送", variant="primary")
                    clear = gr.Button("清空")

                status_line = gr.Textbox(label="状态", value="就绪", interactive=False)
                voice_out = gr.Audio(label="小航语音回复", autoplay=True, interactive=False)

        gr.Markdown("""
        ---
        ### 试试这些指令:
        | 导航 | 车控 | 娱乐 | 安全 | 陪伴 |
        |------|------|------|------|------|
        | 导航到天安门 | 打开空调26度 | 播放周杰伦的歌 | 胎压正常吗 | 我今天心情不好 |
        ---
        *LoongArch 端侧AI · 100%本地推理 · 隐私安全保障*
        """)

        # 事件绑定 (Gradio 6.0) — 语音选择+音高+语速全部由滑块直驱
        send.click(fn=process_text, inputs=[txt, chatbot, voice_selector, pitch_slider, rate_slider],
                   outputs=[chatbot, status_line, voice_out]).then(
                   lambda: "", outputs=[txt])
        txt.submit(fn=process_text, inputs=[txt, chatbot, voice_selector, pitch_slider, rate_slider],
                   outputs=[chatbot, status_line, voice_out]).then(
                   lambda: "", outputs=[txt])
        mic.stop_recording(fn=process_audio, inputs=[mic, chatbot, voice_selector, pitch_slider, rate_slider],
                           outputs=[chatbot, status_line, voice_out])

        # 风格预设 → 一键设滑块
        def _apply_preset(preset_name):
            p, r = VOICE_STYLE_PRESETS.get(preset_name, ("+0Hz", "+0%"))
            return int(p.replace("Hz", "").replace("+", "")), int(r.replace("%", "").replace("+", ""))
        voice_style.change(fn=_apply_preset, inputs=[voice_style],
                           outputs=[pitch_slider, rate_slider])
        clear.click(fn=lambda: ([], "已清空", None),
                    outputs=[chatbot, status_line, voice_out])

        # 表情状态轮询
        emo_timer = gr.Timer(0.5)
        emo_timer.tick(fn=get_emotion_status,
                       outputs=[emotion_display, emotion_bar, status_box])

        # 后台 TTS 音频推送
        audio_timer = gr.Timer(0.3)
        audio_timer.tick(fn=get_pending_audio, outputs=[voice_out])

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True,
                theme=gr.themes.Soft(), show_error=True)
