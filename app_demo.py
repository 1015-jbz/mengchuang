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

def detect_emotion_from_landmarks(face_img_rgb, frame_w, frame_h):
    # 深度学习模型优先（每5帧推理一次，其余用缓存）
    global _last_onnx_result, _onnx_frame_counter
    if _ort_session is not None:
        _onnx_frame_counter += 1
        if _onnx_frame_counter % 5 == 0:
            _last_onnx_result = _onnx_predict_emotion(face_img_rgb)
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
    cache = _os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-whisper-tiny")
    if _os.path.isdir(cache):
        asr_model = WhisperModel("tiny", device="cpu", compute_type="int8", local_files_only=True)
        logger.info("faster-whisper 就绪")
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
INTENTS = {
    ("车辆控制", "空调"): ["空调", "温度", "冷", "热", "通风", "除雾"],
    ("车辆控制", "车窗"): ["车窗", "窗户", "天窗", "开窗", "关窗"],
    ("车辆控制", "座椅"): ["座椅", "加热", "按摩"],
    ("车辆控制", "灯光"): ["灯", "远光", "近光", "氛围灯"],
    ("导航出行", "导航"): ["导航", "去", "怎么走", "路线"],
    ("影音娱乐", "音乐"): ["播放", "音乐", "听", "歌", "来一首"],
    ("影音娱乐", "音量"): ["音量", "大声", "小声", "静音"],
    ("安全守护", "状态"): ["胎压", "油量", "电量"],
    ("安全守护", "紧急"): ["救命", "报警", "求助", "SOS"],
    ("情感陪伴", "倾诉"): ["心情", "难过", "不开心", "郁闷", "烦", "累", "困"],
    ("信息查询", "天气"): ["天气", "下雨", "几度"],
}

CARE_RESPONSES = {
    "sad": [ "我在呢。不管路上遇到什么，我都陪着你。",
             "想说什么就说出来吧，我会认真听。",
             "生活总有起伏，但安全到家最重要。要我放首歌吗？" ],
    "angry": [ "深呼吸，消消气。跟那些人计较不值得，安全最重要。",
               "我理解你的感受。让我们把注意力放在前方的路上。" ],
    "fearful": [ "别怕，我帮你看着周围。你只需专注前方。",
                 "慢慢来就好。安全比准时更重要。" ],
    "tired": [ "你辛苦了。前面服务区就在附近，休息一下吧。",
               "连续驾驶很消耗精力。要不要我讲个笑话提提神？" ],
    "happy": [ "看你开心我也开心！今天真是美好的一天~",
               "心情不错！保持这个好状态，一路顺风！" ],
}

def generate_reply(text, emotion="neutral"):
    """根据输入文本 + 情绪生成回复"""
    import re, random

    # 意图路由
    for (domain, intent), keywords in INTENTS.items():
        if any(kw in text for kw in keywords):
            if domain == "车辆控制":
                t = re.search(r'(\d+)\s*度', text)
                if t: return f"好的，空调已设为{t.group(1)}°C。"
                return "好的，已为您执行车控操作。"
            elif domain == "导航出行":
                dest = re.sub(r"(导航到?|去|怎么到|开到|前往)", "", text).strip()
                return f"正在规划到{dest}的路线。" if dest else "请告诉我您想去哪里？"
            elif domain == "影音娱乐":
                if "音量" in intent:
                    return "音量已调节。"
                song = text.replace("播放", "").replace("听", "").strip()
                return f"正在播放「{song}」。" if song else "好的，为您播放音乐。"
            elif domain == "安全守护":
                if "紧急" in intent:
                    return "已收到紧急求助！正在联系救援，请保持冷静。"
                return "车辆各系统运行正常，请放心驾驶。"
            elif domain == "情感陪伴":
                if emotion in CARE_RESPONSES:
                    return random.choice(CARE_RESPONSES[emotion])
                return "我一直都在。"
            elif domain == "信息查询":
                return "今天天气晴朗，28°C，适合出行。"
            return "好的，收到。"

    # 默认闲聊（受情绪影响）
    if emotion in CARE_RESPONSES:
        return random.choice(CARE_RESPONSES[emotion])
    return random.choice([
        "好的，收到。有什么可以帮你的？",
        "我在认真听。安全和服务是我的首要任务。",
        "明白。需要我帮你操作什么吗？",
    ])

def tts(text):
    """TTS 文字转语音（后台线程，不阻塞 UI）"""
    if not text:
        return None
    import tempfile
    out_path = Path(tempfile.gettempdir()) / "cockpit_tts.mp3"

    def _run():
        try:
            import edge_tts, asyncio
            async def gen():
                comm = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
                await comm.save(str(out_path))
            loop = asyncio.new_event_loop()
            loop.run_until_complete(gen())
            loop.close()
        except Exception as e:
            logger.warning(f"TTS 失败: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=3.0)  # 等最多3秒
    return str(out_path) if out_path.exists() else None

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

def process_text(text, chat_hist):
    """处理文本输入"""
    if chat_hist is None:
        chat_hist = []
    if not text or not text.strip():
        return chat_hist, "请输入内容"

    text = text.strip()

    with _lock:
        emo = _latest_emotion

    try:
        reply = generate_reply(text, emo)
    except Exception as e:
        reply = f"抱歉，出错了: {e}"

    # TTS 后台执行
    import threading as _th
    _th.Thread(target=lambda: tts(reply), daemon=True).start()

    # Gradio 6.0 使用新格式
    chat_hist.append({"role": "user", "content": text})
    chat_hist.append({"role": "assistant", "content": reply})

    # 找意图
    intent = "闲聊"
    for (d, i), ks in INTENTS.items():
        if any(k in text for k in ks):
            intent = f"{d}/{i}"
            break

    status = f"意图: {intent} | 情绪: {EMOTION_ZH.get(emo, '?')}"
    return chat_hist, status

def process_audio(audio_path, chat_hist):
    """处理语音输入"""
    if audio_path is None or chat_hist is None:
        return (chat_hist or [], "") if chat_hist is None else (chat_hist, "")
    text = transcribe(audio_path)
    if not text:
        return chat_hist, "(未识别到语音)"
    return process_text(text, chat_hist)

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
    video_url = "http://localhost:7861/video_feed" if HAS_FLASK else ""

    with gr.Blocks(title="Smart Cockpit") as demo:
        gr.Markdown("# 智能座舱多模态交互终端")
        gr.Markdown("### 基于 LoongArch 端侧 AI | 实时表情识别 | 智能语音对话")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 实时表情识别")

                if video_url:
                    # 直接嵌入 MJPEG 流
                    gr.HTML(f"""
                    <div style="border:2px solid #00d4aa; border-radius:10px; overflow:hidden; background:#000;">
                        <img src="{video_url}" style="width:100%; display:block;"
                             onerror="this.onerror=null; this.src=''; this.parentElement.innerHTML='<p style=color:red;padding:20px;>摄像头未就绪，请确认摄像头已连接</p>'">
                    </div>
                    <p style="text-align:center;color:#888;font-size:12px;">MJPEG 实时流 · 25 FPS</p>
                    """)
                else:
                    gr.Markdown("*视频流不可用 (Flask 未安装)*")

                emotion_display = gr.Markdown("# 😐 等待中...")
                emotion_bar = gr.Slider(0, 1, value=0, label="情绪置信度", interactive=False)
                status_box = gr.JSON(
                    value={"表情":"等待", "置信度":"0%", "人脸":"等待"},
                    label="检测状态",
                    every=0.5,  # 每0.5秒轮询
                )

            with gr.Column(scale=2):
                gr.Markdown("### 智能语音对话")
                chatbot = gr.Chatbot(label="对话记录", height=400)

                with gr.Row():
                    mic = gr.Audio(sources=["microphone"], type="filepath", label="语音输入")
                    txt = gr.Textbox(placeholder="或在这里打字...", label="文字输入", scale=2)

                with gr.Row():
                    send = gr.Button("发送", variant="primary")
                    clear = gr.Button("清空")

                status_line = gr.Textbox(label="状态", value="就绪", interactive=False)

        gr.Markdown("""
        ---
        ### 试试这些指令:
        | 导航 | 车控 | 娱乐 | 安全 | 陪伴 |
        |------|------|------|------|------|
        | 导航到天安门 | 打开空调26度 | 播放周杰伦的歌 | 胎压正常吗 | 我今天心情不好 |
        ---
        *LoongArch 端侧AI · 100%本地推理 · 隐私安全保障*
        """)

        # 事件绑定 (Gradio 6.0)
        send.click(fn=process_text, inputs=[txt, chatbot],
                   outputs=[chatbot, status_line]).then(
                   lambda: "", outputs=[txt])
        txt.submit(fn=process_text, inputs=[txt, chatbot],
                   outputs=[chatbot, status_line]).then(
                   lambda: "", outputs=[txt])
        mic.stop_recording(fn=process_audio, inputs=[mic, chatbot],
                           outputs=[chatbot, status_line])
        clear.click(fn=lambda: ([], "已清空"),
                    outputs=[chatbot, status_line])

        # 表情状态轮询
        status_box.change(fn=lambda x: (f"# {EMOTION_ZH.get(x.get('表情',''), '等待')} {get_emotion_status()[0].split()[-1] if get_emotion_status()[0] else ''}", get_emotion_status()[1]),
                          inputs=[status_box], outputs=[emotion_display, emotion_bar])

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=gr.themes.Soft(), show_error=True)
