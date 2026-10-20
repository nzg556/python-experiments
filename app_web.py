# -*- coding: utf-8 -*-
"""
模块（6）：部署与应用 —— Web 端情绪识别系统
====================================
对应实训步骤 6：将训练好的 CNN 模型部署到实际应用中，
集成到 Web 应用程序，实现实时的情绪识别功能。

功能：
    1. 静态识别：浏览器上传图片 -> 服务端人脸检测 + CNN 识别 -> 返回标注图与结果；
    2. 实时识别：浏览器调用摄像头(getUserMedia) -> 逐帧提交服务端识别 -> 画面实时叠加结果。

运行：
    python app_web.py                # 默认 http://127.0.0.1:5000
    python app_web.py --port 8000    # 指定端口

浏览器访问：
    http://127.0.0.1:5000            静态图片识别
    http://127.0.0.1:5000/realtime   摄像头实时识别

说明：
    - 摄像头画面由浏览器采集（getUserMedia），服务端无需占用摄像头，
      因此 detect_webcam.py 与本 Web 应用可独立使用；
    - getUserMedia 仅在 localhost 或 HTTPS 下可用，本机演示直接可用。
"""
import argparse
import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import config
from face_detector import (FaceDetector, crop_square_face, draw_result,
                           load_emotion_model, predict_emotion)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 上限 16MB

# 模型与人脸检测器在启动时加载一次（全局共享，线程安全用于演示场景）
_model = None
_detector = None


def init_services():
    """加载表情识别模型与人脸检测器。"""
    global _model, _detector
    if _model is None:
        _model = load_emotion_model()
        print("表情识别模型已加载:", config.MODEL_PATH)
    if _detector is None:
        _detector = FaceDetector()


# ---------------------------------------------------------------- 工具函数
def _recognize_image(image_bgr, annotate=True):
    """
    对一张 BGR 图像完成 人脸检测 -> 表情识别。

    返回:
        results: [{box, label, label_zh, confidence}, ...]
        annotated: 标注后的图像（annotate=False 时为 None）
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    boxes = _detector.detect(image_bgr)
    results = []

    if not boxes:
        # 未检测到人脸：对整张图直接预测一次（与 detect_gui 行为一致）
        idx, conf, _ = predict_emotion(_model, gray)
        results.append({
            "box": None,
            "label": config.EMOTION_LABELS[idx],
            "label_zh": config.EMOTION_LABELS_ZH[idx],
            "confidence": round(conf, 4),
            "whole_image": True,
        })
        return results, None

    annotated = image_bgr.copy() if annotate else None
    for box in boxes:
        face = crop_square_face(gray, box)
        if face.size == 0:
            continue
        idx, conf, _ = predict_emotion(_model, face)
        results.append({
            "box": [int(v) for v in box],
            "label": config.EMOTION_LABELS[idx],
            "label_zh": config.EMOTION_LABELS_ZH[idx],
            "confidence": round(conf, 4),
            "whole_image": False,
        })
        if annotate:
            draw_result(annotated, box, idx, conf, use_chinese=True)
    return results, annotated


def _to_data_url(image_bgr):
    """BGR 图像 -> JPEG base64 data URL，用于网页显示。"""
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _decode_data_url(data_url):
    """浏览器提交的 base64 data URL -> BGR 图像。"""
    try:
        payload = data_url.split(",", 1)[1] if "," in data_url else data_url
        buf = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------- 页面路由
@app.route("/")
def index():
    """静态图片识别页面。"""
    return render_template("index.html")


@app.route("/realtime")
def realtime():
    """摄像头实时识别页面。"""
    return render_template("realtime.html")


@app.route("/health")
def health():
    """健康检查接口。"""
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "classes": config.EMOTION_LABELS,
    })


# ---------------------------------------------------------------- 识别接口
@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    """静态识别接口：接收上传图片，返回标注图与识别结果。"""
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="请先选择一张图片再提交。")

    image = cv2.imdecode(np.frombuffer(file.read(), dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if image is None:
        return render_template("index.html", error="图片读取失败，请更换图片格式（jpg/png/bmp）。")

    results, annotated = _recognize_image(image, annotate=True)
    result_image = _to_data_url(annotated) if annotated is not None else None
    return render_template("index.html", results=results, result_image=result_image)


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """实时识别接口：接收浏览器采集的一帧 base64 图像，返回人脸框与表情。"""
    data = request.get_json(silent=True) or {}
    image = _decode_data_url(data.get("image", ""))
    if image is None:
        return jsonify({"error": "图像解码失败"}), 400

    results, _ = _recognize_image(image, annotate=False)
    return jsonify({"results": results, "count": len(results)})


# ---------------------------------------------------------------- 入口
def main():
    parser = argparse.ArgumentParser(description="Web 端情绪识别系统（模块6 部署与应用）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    args = parser.parse_args()

    init_services()
    print(f"启动 Web 服务: http://{args.host}:{args.port}")
    # 局域网内其他设备访问时使用 --host 0.0.0.0
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
