"""
LBP 表情识别引擎 — 局部二值模式 + 卡方距离
替换 app_demo.py 中对应的表情检测函数
"""
import cv2
import numpy as np
import logging
logger = logging.getLogger(__name__)


def _lbp_histogram(roi, bins=64):
    """计算 LBP 直方图（OpenCV 原生，无需额外依赖）"""
    if roi.size == 0:
        return np.zeros(bins, dtype=np.float32)
    lbp = np.zeros_like(roi, dtype=np.uint8)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        shifted = np.roll(np.roll(roi, dy, axis=0), dx, axis=1)
        lbp += (shifted >= roi).astype(np.uint8)
    hist = cv2.calcHist([lbp], [0], None, [bins], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _chi2_distance(h1, h2):
    """卡方距离"""
    h1, h2 = h1.astype(np.float64), h2.astype(np.float64)
    denom = h1 + h2 + 1e-10
    return float(np.sum((h1 - h2) ** 2 / denom))


def extract_lbp_features(face_rgb):
    """提取嘴部/眉心/眼部三个区域的 LBP 纹理特征（192维）"""
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    mouth = _lbp_histogram(gray[2*h//3:, w//6:5*w//6])
    glabella = _lbp_histogram(gray[h//8:3*h//8, w//3:2*w//3])
    eye = _lbp_histogram(gray[h//6:h//3, :])
    return np.concatenate([mouth, glabella, eye])


def baseline_update(features, baseline, count):
    """指数移动平均更新基线"""
    if baseline is None:
        return features.copy()
    w = count / (count + 1)
    return baseline * w + features * (1 - w)


def classify_emotion_lbp(features, baseline, smile_cascade, face_rgb, total_min=0.12):
    """
    基于 LBP 直方图距离的表情分类。

    原理：不同表情改变不同面部区域的纹理
    - 微笑 → 嘴部 LBP 大变 + smile cascade 确认
    - 惊讶 → 嘴部 + 眼部 LBP 都大变
    - 愤怒 → 眉心 LBP 变大（主）+ 嘴部变化小
    - 悲伤 → 全局变化小 + 嘴部尤其小
    - 恐惧 → 眼部 LBP 大 + 嘴部中等

    total_min: 总距离低于此值视为平静脸，避免微小抖动误判
    """
    n = len(features) // 3
    mouth_feat = features[:n]
    glabella_feat = features[n:2*n]
    eye_feat = features[2*n:]

    mouth_dist = _chi2_distance(mouth_feat, baseline[:n])
    glabella_dist = _chi2_distance(glabella_feat, baseline[n:2*n])
    eye_dist = _chi2_distance(eye_feat, baseline[2*n:])

    total = mouth_dist + glabella_dist + eye_dist

    # 总变化太小 → 平静
    if total < total_min:
        return "neutral", 0.5

    # 😊 开心 — smile cascade（训练过的模型，置信度高）
    if smile_cascade is not None:
        h, w = face_rgb.shape[:2]
        gray_lower = cv2.cvtColor(face_rgb[h//2:, :], cv2.COLOR_RGB2GRAY)
        for sf, mn in [(1.3, 15), (1.5, 10), (1.8, 8)]:
            smiles = smile_cascade.detectMultiScale(
                gray_lower, sf, mn, minSize=(20, 15))
            if len(smiles) > 0:
                return "happy", 0.8

    m_pct = mouth_dist / max(total, 0.001)
    g_pct = glabella_dist / max(total, 0.001)
    e_pct = eye_dist / max(total, 0.001)

    # 😲 惊讶 — 嘴部+眼部都大变（张嘴+眼睛睁大）
    if m_pct > 0.35 and e_pct > 0.15:
        return "surprised", min(0.9, total * 1.2)

    # 😤 愤怒 — 眉心区域变化占主导
    if g_pct > 0.3 and m_pct < 0.5:
        return "angry", min(0.85, total * 1.0)

    # 😨 恐惧 — 眼部变化占主导 + 嘴部也有变化
    if e_pct > 0.3 and m_pct > 0.2:
        return "fearful", 0.6

    # 😢 悲伤 — 整体有变化但嘴部变化小
    if m_pct < 0.25 and e_pct < 0.25 and g_pct > 0.15:
        return "sad", 0.55

    return "neutral", 0.5
