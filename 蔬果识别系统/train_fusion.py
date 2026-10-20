# -*- coding: utf-8 -*-
"""融合训练（全量131类）：CNN softmax 概率 + 颜色/形状特征 -> LogisticRegression
训练特征每类采样 120 张（131×120=15720 张），测试集使用全部 22688 张评估。
"""
import os
import cv2
import json
import numpy as np
import tensorflow as tf
import joblib
from sklearn.linear_model import LogisticRegression

from features import extract_features

DATA_ROOT = "dataset/fruits-360"
with open("webapp/model/labels.json", encoding="utf-8") as f:
    CLASSES = json.load(f)
N = len(CLASSES)

print("loading CNN model...")
model = tf.keras.models.load_model("webapp/model/fruit_veg_cnn.h5")


def build(split, limit_per_class=None):
    X_fuse, y = [], []
    for idx, c in enumerate(CLASSES):
        d = os.path.join(DATA_ROOT, split, c)
        files = sorted(os.listdir(d))
        if limit_per_class:
            # 均匀抽样，避免只取连续拍摄的图片
            if len(files) > limit_per_class:
                sel = np.linspace(0, len(files) - 1, limit_per_class).astype(int)
                files = [files[i] for i in sel]
        imgs, feats = [], []
        for f in files:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(d, f)), cv2.COLOR_BGR2RGB)
            imgs.append(cv2.resize(rgb, (100, 100)).astype(np.float32) / 255.0)
            feats.append(extract_features(rgb))
        imgs = np.array(imgs)
        # 分批 CNN 推理，避免一次性占满内存
        probs = np.vstack([model.predict(b, verbose=0) for b in np.array_split(imgs, max(1, len(imgs)//64)+1)])
        X_fuse.append(np.concatenate([probs, np.array(feats)], axis=1))
        y.extend([idx] * len(files))
        if (idx + 1) % 20 == 0:
            print(f"  {split}: {idx+1}/{N} 类完成")
    return np.vstack(X_fuse), np.array(y)


print("building train features (每类120张)...")
X_train, y_train = build("Training", limit_per_class=120)
print(f"train fusion matrix: {X_train.shape}")
print("building test features (全量)...")
X_test, y_test = build("Test")
print(f"test fusion matrix: {X_test.shape}")

clf = LogisticRegression(max_iter=3000, C=1.0)
clf.fit(X_train, y_train)
acc_fuse = clf.score(X_test, y_test)
acc_cnn = np.mean(np.argmax(X_test[:, :N], axis=1) == y_test)
print(f"CNN-only test acc      = {acc_cnn:.4f}")
print(f"CNN+颜色/形状融合 acc   = {acc_fuse:.4f}")

joblib.dump(clf, "webapp/model/fusion_clf.joblib")
print("saved webapp/model/fusion_clf.joblib")
