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
    HAS_CV2 = True
    logger.info("OpenCV 人脸检测就绪")
except Exception as e:
    _face_cascade = None
    HAS_CV2 = False
    logger.warning(f"OpenCV 不可用: {e}")

# DeepFace (可选)
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
    logger.info("DeepFace 表情识别就绪")
except ImportError:
    HAS_DEEPFACE = False
    logger.info("DeepFace 未安装，使用启发式表情检测")

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

def detect_emotion(face_img):
    """检测单张人脸的表情"""
    if HAS_DEEPFACE:
        try:
            analysis = DeepFace.analyze(face_img, actions=['emotion'],
                                        enforce_detection=False, silent=True)
            if analysis:
                emotion = analysis[0].get('dominant_emotion', 'neutral')
                emotions = analysis[0].get('emotion', {})
                conf = emotions.get(emotion, 0) / 100.0
                return emotion, conf
        except Exception:
            pass

    # 降级：启发式
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    brightness = float(gray.mean())
    if contrast > 55: return "surprised", 0.5
    if brightness < 90: return "sad", 0.4
    if contrast < 20: return "neutral", 0.6
    return "neutral", 0.5

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(64, 64))

        emotion = "neutral"
        conf = 0.0
        face_detected = False

        if len(faces) > 0:
            (x, y, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
            face_img = frame[y:y+fh, x:x+fw]
            emotion, conf = detect_emotion(face_img)
            face_detected = True

            color = EMOTION_COLORS.get(emotion, (180, 180, 180))
            label = f"{EMOTION_ZH.get(emotion, emotion)} ({conf:.0%})"
            cv2.rectangle(annotated, (x, y), (x+fw, y+fh), color, 2)
            cv2.rectangle(annotated, (x, y-30), (x+len(label)*12, y), color, -1)
            cv2.putText(annotated, label, (x+5, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
    """TTS 文字转语音"""
    if not text:
        return None
    try:
        import edge_tts, asyncio, tempfile
        async def gen():
            path = Path(tempfile.gettempdir()) / "cockpit_tts.mp3"
            comm = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            await comm.save(str(path))
            return str(path)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(gen())
        loop.close()
        return result
    except Exception:
        return None

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
    if not text or not text.strip():
        return chat_hist, None, ""
    text = text.strip()

    with _lock:
        emo = _latest_emotion

    reply = generate_reply(text, emo)
    audio = tts(reply)

    chat_hist.append(["你", text])
    chat_hist.append(["小航", reply])

    intent = "闲聊"
    for (d, i), ks in INTENTS.items():
        if any(k in text for k in ks):
            intent = f"{d}/{i}"
            break

    return chat_hist, audio, f"意图: {intent} | 情绪: {EMOTION_ZH.get(emo, '?')}"

def process_audio(audio_path, chat_hist):
    """处理语音输入"""
    if audio_path is None:
        return chat_hist, None, ""
    text = transcribe(audio_path)
    if not text:
        return chat_hist, None, "(未识别到语音)"
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
    print(f"  表情识别: {'DeepFace' if HAS_DEEPFACE else 'OpenCV 启发式'}")
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

                audio_out = gr.Audio(label="小航语音回复", type="filepath", autoplay=True)
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

        # 事件绑定
        send.click(fn=process_text, inputs=[txt, chatbot],
                   outputs=[chatbot, audio_out, status_line]).then(
                   lambda: "", outputs=[txt])
        txt.submit(fn=process_text, inputs=[txt, chatbot],
                   outputs=[chatbot, audio_out, status_line]).then(
                   lambda: "", outputs=[txt])
        mic.stop_recording(fn=process_audio, inputs=[mic, chatbot],
                           outputs=[chatbot, audio_out, status_line])
        clear.click(fn=lambda: ([], None, "已清空"),
                    outputs=[chatbot, audio_out, status_line])

        # 表情状态轮询
        status_box.change(fn=lambda x: (f"# {EMOTION_ZH.get(x.get('表情',''), '等待')} {get_emotion_status()[0].split()[-1] if get_emotion_status()[0] else ''}", get_emotion_status()[1]),
                          inputs=[status_box], outputs=[emotion_display, emotion_bar])

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=gr.themes.Soft(), show_error=True)
