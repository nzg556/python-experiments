# -*- coding: utf-8 -*-
"""颜色与形状特征提取模块（CNN 融合用）
fruits-360 图像为白色背景，可通过阈值分割出蔬果主体，
提取 HSV 颜色直方图与轮廓形状特征。
"""
import cv2
import numpy as np

IMG_SIZE = (100, 100)


def extract_features(img_rgb):
    """输入 RGB uint8 图像，返回 1D 特征向量（颜色 + 形状）"""
    img = cv2.resize(img_rgb, IMG_SIZE)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # --- 白色背景分割：主体为饱和度较高或较暗的像素 ---
    mask = ((hsv[:, :, 1] > 40) | (hsv[:, :, 2] < 200)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # ---------- 颜色特征 ----------
    # 主体区域 H 直方图（32  bins）
    h_hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [8], [0, 256]).flatten()
    total = h_hist.sum() + 1e-6
    h_hist = h_hist / total
    s_hist = s_hist / (s_hist.sum() + 1e-6)

    # 主体平均颜色（RGB / HSV 各通道均值）
    mean_rgb = cv2.mean(img, mask=mask)[:3]
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]

    # ---------- 形状特征 ----------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)

        area_ratio = area / (IMG_SIZE[0] * IMG_SIZE[1])        # 主体面积占比（大小）
        circularity = 4 * np.pi * area / (peri ** 2 + 1e-6)     # 圆形度
        aspect = w / (h + 1e-6)                                 # 长宽比
        extent = area / (w * h + 1e-6)                          # 矩形填充度
        solidity = area / (hull_area + 1e-6)                    # 凸包实心度
    else:
        area_ratio = circularity = aspect = extent = solidity = 0.0

    shape = np.array([area_ratio, circularity, aspect, extent, solidity], dtype=np.float32)

    return np.concatenate([h_hist, s_hist,
                           np.array(mean_rgb, dtype=np.float32) / 255.0,
                           np.array(mean_hsv, dtype=np.float32) / np.array([180., 255., 255.], dtype=np.float32),
                           shape]).astype(np.float32)
