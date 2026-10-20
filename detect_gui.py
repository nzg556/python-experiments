# -*- coding: utf-8 -*-
"""
模块（4）：静态图片情绪识别（Tkinter 图形界面）
====================================
对应实训指导书“（4）静态检测”。

功能：
    1. 点击“选择图片”选取本地图片；
    2. 自动检测图片中的人脸并用 CNN 识别表情；
    3. 在界面中绘制人脸框与中文结果，同时弹窗提示；
    4. 若图片中未检测到人脸，则对整张图片直接识别一次。

用法：
    python detect_gui.py
"""
import tkinter as tk
import tkinter.filedialog
from tkinter.messagebox import showinfo, showerror

import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from face_detector import (FaceDetector, crop_square_face, draw_result,
                           load_emotion_model, predict_emotion)

# 界面中图片显示的最大边长
MAX_DISPLAY_SIZE = 480


class EmotionRecognitionApp:
    """表情识别主窗口。"""

    def __init__(self, model, detector):
        self.model = model
        self.detector = detector

        self.app = tk.Tk()
        self.app.title("人脸表情识别")
        self.app.geometry("720x640+200+120")

        # 图片路径
        self.path = tk.StringVar()
        entry = tk.Entry(self.app, state="readonly", text=self.path,
                         width=80, font=("Microsoft YaHei", 10))
        entry.pack(pady=8)

        # 操作按钮
        button_frame = tk.Frame(self.app)
        button_frame.pack()
        tk.Button(button_frame, text="选择图片", width=14,
                  command=self.choose_image).pack(side=tk.LEFT, padx=6)
        tk.Button(button_frame, text="退出", width=14,
                  command=self.app.destroy).pack(side=tk.LEFT, padx=6)

        # 图片显示区域
        self.image_label = tk.Label(self.app, bg="#dddddd")
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 识别结果文字
        self.result_var = tk.StringVar(value="请选择一张人脸图片进行识别")
        tk.Label(self.app, textvariable=self.result_var,
                 font=("Microsoft YaHei", 13), fg="blue").pack(pady=6)

        # 保持图片引用，防止被垃圾回收导致界面不显示
        self._photo = None

    def choose_image(self):
        """选择图片 -> 检测人脸 -> 表情识别 -> 显示结果。"""
        file_path = tkinter.filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                       ("所有文件", "*.*")],
        )
        if not file_path:
            return

        self.path.set(file_path)

        # cv2.imread 不支持中文路径，改用 np.fromfile + imdecode
        image = self._imread_unicode(file_path)
        if image is None:
            showerror("错误", "无法读取该图片，请更换图片。")
            return

        result_text, annotated = self._recognize(image)
        self.result_var.set(result_text)

        # 在界面中显示标注后的图片
        self._show_image(annotated)

        # 与指导书一致，弹窗提示结果
        showinfo("识别结果", result_text)

    def _recognize(self, image_bgr):
        """检测人脸并识别表情，返回 (结果文字, 标注后的图片)。"""
        annotated = image_bgr.copy()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detect(image_bgr)

        if not boxes:
            # 未检测到人脸：对整张图片直接识别一次
            label_index, confidence, _ = predict_emotion(self.model, gray)
            label_zh = config.EMOTION_LABELS_ZH[label_index]
            return (f"未检测到人脸，整图识别结果：{label_zh}"
                    f"（{confidence:.2f}）"), annotated

        texts = []
        for box in boxes:
            face = crop_square_face(gray, box)
            if face.size == 0:
                continue
            label_index, confidence, _ = predict_emotion(self.model, face)
            draw_result(annotated, box, label_index, confidence,
                        use_chinese=True)
            texts.append(f"{config.EMOTION_LABELS_ZH[label_index]}"
                         f"({confidence:.2f})")

        return f"识别结果：{', '.join(texts)}", annotated

    @staticmethod
    def _imread_unicode(file_path):
        """兼容中文路径的图片读取。"""
        try:
            data = np.fromfile(file_path, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except OSError:
            return None

    def _show_image(self, image_bgr):
        """将 BGR 图像等比缩放后显示到 Tkinter Label。"""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE))
        self._photo = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self._photo, width=MAX_DISPLAY_SIZE,
                                   height=MAX_DISPLAY_SIZE)

    def run(self):
        self.app.mainloop()


def main():
    model = load_emotion_model()
    detector = FaceDetector()
    app = EmotionRecognitionApp(model, detector)
    app.run()


if __name__ == "__main__":
    main()
