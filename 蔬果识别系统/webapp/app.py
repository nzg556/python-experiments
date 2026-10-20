# -*- coding: utf-8 -*-
"""蔬果识别 Web 应用（Flask + CNN）
按照《神经网络与深度学习实训》指导书结构组织：
- 静态检测：上传图片识别
- 动态监测：摄像头实时识别
"""
import os
import sys
import json
import base64
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO

# 特征提取模块在上一级目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import extract_features

app = Flask(__name__)

MODEL_PATH = "model/fruit_veg_cnn.h5"
LABELS_PATH = "model/labels.json"
FUSION_PATH = "model/fusion_clf.joblib"

CN_NAMES = {}
KIND = {}

model = tf.keras.models.load_model(MODEL_PATH)
clf = joblib.load(FUSION_PATH)
with open(LABELS_PATH, encoding="utf-8") as f:
    CLASSES = json.load(f)
with open("model/labels_meta.json", encoding="utf-8") as f:
    META = json.load(f)
    CN_NAMES = {k: v["cn"] for k, v in META.items()}
    KIND = {k: v["kind"] for k, v in META.items()}


def predict_bytes(file_bytes):
    """融合识别：CNN + 颜色/形状特征 -> LogisticRegression 融合分类器"""
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)

    # CNN softmax 概率
    x = tf.convert_to_tensor(arr, dtype=tf.float32)
    x = tf.image.resize(x, [100, 100]) / 255.0
    x = tf.expand_dims(x, axis=0)
    cnn_probs = model.predict(x, verbose=0)[0]

    # 颜色与形状特征
    feat = extract_features(arr.astype(np.uint8)).reshape(1, -1)

    # 融合特征输入融合分类器
    fusion = np.hstack([cnn_probs.reshape(1, -1), feat])
    clf_proba = clf.predict_proba(fusion)[0]
    top3 = clf_proba.argsort()[::-1][:3]
    return [{
        "label": CLASSES[i],
        "cn": CN_NAMES.get(CLASSES[i], CLASSES[i]),
        "kind": KIND.get(CLASSES[i], ""),
        "prob": round(float(clf_proba[i]) * 100, 2),
    } for i in top3]


# ---------------- 页面路由 ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static")
def static_detect():
    return render_template("static.html")


@app.route("/dynamic")
def dynamic_detect():
    return render_template("dynamic.html")


@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html")


# ---------------- API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" in request.files and request.files["file"].filename:
        file_bytes = request.files["file"].read()
    elif request.is_json and request.json.get("image"):
        b64 = request.json["image"].split(",", 1)[-1]
        file_bytes = base64.b64decode(b64)
    else:
        return jsonify({"error": "未收到图片"}), 400

    return jsonify({"results": predict_bytes(file_bytes)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
