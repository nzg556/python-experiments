# -*- coding: utf-8 -*-
"""
人脸检测与表情识别公共工具
====================================
将指导书“动态监测”中的人脸检测、人脸区域裁剪、图像预处理、
中文结果绘制等逻辑抽取为可复用函数，供摄像头识别和静态图片识别共用。

人脸检测策略（自动选择，无需任何手动下载 / 硬编码路径）：
    1. 优先使用 OpenCV 自带 DNN 人脸检测器（opencv_face_detector）；
    2. 若当前 opencv 版本未附带 DNN 模型，则回退到 OpenCV 自带的
       Haar 级联检测器（haarcascade_frontalface_default.xml）。
"""
import os

import cv2
import numpy as np

import config


# ---------------------------------------------------------------- 模型加载
def load_emotion_model(model_path=None):
    """加载训练好的表情识别模型。"""
    from tensorflow import keras

    model_path = model_path or config.MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"未找到模型文件 {model_path}，请先运行: python train.py"
        )
    return keras.models.load_model(model_path)


# ---------------------------------------------------------------- 人脸检测
class FaceDetector:
    """人脸检测器：优先 DNN，不可用时回退 Haar 级联。"""

    def __init__(self, conf_threshold=None):
        self.conf_threshold = conf_threshold or config.FACE_CONF_THRESHOLD
        self.backend = None
        self.net = None
        self.cascade = None

        self._init_dnn()
        if self.net is None:
            self._init_cascade()

        if self.net is None and self.cascade is None:
            raise RuntimeError("未找到任何人脸检测器（DNN / Haar 均不可用）。")

        print(f"人脸检测器后端: {self.backend}")

    def _init_dnn(self):
        """尝试加载 OpenCV DNN 人脸检测器。"""
        # 候选位置：opencv 包目录 data/ 与项目 models/ 目录
        cv2_data_dir = os.path.join(os.path.dirname(cv2.__file__), "data")
        candidates = [
            (os.path.join(cv2_data_dir, "opencv_face_detector_uint8.pb"),
             os.path.join(cv2_data_dir, "opencv_face_detector.pbtxt")),
            (os.path.join(config.MODEL_DIR, "opencv_face_detector_uint8.pb"),
             os.path.join(config.MODEL_DIR, "opencv_face_detector.pbtxt")),
        ]
        for model_file, config_file in candidates:
            if os.path.exists(model_file) and os.path.exists(config_file):
                try:
                    self.net = cv2.dnn.readNetFromTensorflow(
                        model_file, config_file)
                    self.backend = "DNN (opencv_face_detector)"
                    return
                except cv2.error:
                    self.net = None

    def _init_cascade(self):
        """回退：加载 Haar 级联检测器（opencv-python 默认自带）。"""
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        if os.path.exists(cascade_path):
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if not self.cascade.empty():
                self.backend = "Haar Cascade (haarcascade_frontalface_default)"

    def detect(self, frame_bgr):
        """
        检测 BGR 图像中的人脸。

        返回:
            [(x1, y1, x2, y2), ...]  人脸矩形列表
        """
        if self.net is not None:
            return self._detect_dnn(frame_bgr)
        return self._detect_cascade(frame_bgr)

    def _detect_dnn(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr, 1.0, (300, 300), [104, 117, 123], False, False
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.conf_threshold:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _detect_cascade(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        return [(int(x), int(y), int(x + w), int(y + h))
                for (x, y, w, h) in faces]


# ---------------------------------------------------------------- 人脸裁剪
def crop_square_face(frame_gray, box):
    """
    将检测框扩展为正方形后裁剪人脸区域（对应指导书 get_face 中的逻辑），
    避免直接 resize 造成形变。
    """
    h, w = frame_gray.shape[:2]
    x1, y1, x2, y2 = box
    face_h, face_w = y2 - y1, x2 - x1
    side = max(face_w, face_h)

    # 以检测框中心为中心向四周扩展为正方形
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    nx1 = cx - side // 2
    ny1 = cy - side // 2
    nx2, ny2 = nx1 + side, ny1 + side

    # 边界平移修正
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > w:
        nx1 -= (nx2 - w)
        nx2 = w
    if ny1 > h:
        ny1 -= (ny2 - h)
        ny2 = h
    nx1, ny1 = max(0, nx1), max(0, ny1)

    return frame_gray[ny1:ny2, nx1:nx2]


# ---------------------------------------------------------------- 预处理/预测
def preprocess_face(face_gray):
    """
    人脸灰度图 -> 模型输入。
    返回 shape=(1, 100, 100, 1)、取值 0~1 的 float32 数组。
    """
    img = cv2.resize(face_gray, (config.IMG_SIZE, config.IMG_SIZE),
                     interpolation=cv2.INTER_AREA)
    x = img.astype("float32") / 255.0
    x = x.reshape(1, config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS)
    return x


def predict_emotion(model, face_gray):
    """
    对单张人脸灰度图进行表情预测。

    返回:
        label_index: int
        confidence: float   预测类别的概率
        probabilities: np.ndarray  7 类概率
    """
    x = preprocess_face(face_gray)
    probabilities = model.predict(x, verbose=0)[0]
    label_index = int(np.argmax(probabilities))
    return label_index, float(probabilities[label_index]), probabilities


# ---------------------------------------------------------------- 结果绘制
_FONT_CACHE = {}


def _get_chinese_font(size):
    """加载中文字体（带缓存），找不到则返回 None。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont

    font = None
    for font_path in config.FONT_CANDIDATES:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, size)
                break
            except OSError:
                continue
    _FONT_CACHE[size] = font
    return font


def draw_result(frame_bgr, box, label_index, confidence=None, use_chinese=True):
    """
    在帧上绘制人脸矩形框与表情结果。优先使用 PIL 绘制中文，
    字体不可用时自动回退 OpenCV 英文文字。
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if use_chinese:
        text = config.EMOTION_LABELS_ZH[label_index]
        if confidence is not None:
            text += f" {confidence:.2f}"
        ok = _draw_chinese_text(frame_bgr, text, (x1, max(0, y1 - 38)))
        if ok:
            return frame_bgr

    # 回退：OpenCV 英文标签
    text = config.EMOTION_LABELS[label_index]
    if confidence is not None:
        text += f" {confidence:.2f}"
    cv2.putText(frame_bgr, text, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return frame_bgr


def _draw_chinese_text(frame_bgr, text, pos):
    """使用 PIL 在 OpenCV 图像（BGR）上绘制中文，成功返回 True。"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    font_size = max(22, frame_bgr.shape[1] // 28)
    font = _get_chinese_font(font_size)
    if font is None:
        return False

    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # PIL 中 RGB=(0, 225, 225) 对应蓝色；加描边保证在亮背景上也清晰
    draw.text(pos, text, font=font, fill=(0, 225, 225),
              stroke_width=1, stroke_fill=(0, 0, 0))
    frame_bgr[:] = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    return True
