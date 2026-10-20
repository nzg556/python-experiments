# -*- coding: utf-8 -*-
"""
模块（3）：摄像头动态情绪识别
====================================
对应实训指导书“（3）动态监测”。

打开摄像头实时检测人脸，并对每张人脸用训练好的 CNN 模型进行表情分类，
将人脸框与识别结果（中文 / 英文）实时绘制到画面上。

用法：
    python detect_webcam.py
    python detect_webcam.py --english        # 强制使用英文标签
    python detect_webcam.py --camera 0       # 指定摄像头编号

按 q 键退出。
"""
import argparse

import cv2

import config
from face_detector import (FaceDetector, crop_square_face, draw_result,
                           load_emotion_model, predict_emotion)


def main():
    parser = argparse.ArgumentParser(description="摄像头实时表情识别")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--english", action="store_true",
                        help="使用英文标签显示结果")
    parser.add_argument("--width", type=int, default=800, help="画面宽度")
    parser.add_argument("--height", type=int, default=600, help="画面高度")
    args = parser.parse_args()

    # 加载模型与人脸检测器
    model = load_emotion_model()
    detector = FaceDetector()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头（编号 {args.camera}），"
                           "请检查设备连接或尝试 --camera 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("摄像头已开启，按 q 键退出...")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取摄像头画面失败。")
            break

        # 水平翻转，获得镜像视角（与指导书一致）
        frame = cv2.flip(frame, flipCode=1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测当前帧中的所有人脸
        boxes = detector.detect(frame)
        for box in boxes:
            face = crop_square_face(gray, box)
            if face.size == 0:
                continue
            label_index, confidence, _ = predict_emotion(model, face)
            draw_result(frame, box, label_index, confidence,
                        use_chinese=not args.english)

        cv2.putText(frame, "q: quit", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
