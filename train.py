# -*- coding: utf-8 -*-
"""
模块（2）：模型训练与评估
====================================
对应实训指导书“（2）模型训练”。

流程：
    1. 加载 prepare_dataset.py 生成的 data.npz；
    2. 像素归一化到 0~1；
    3. 构建 CNN 模型并训练（SGD + 早停）；
    4. 在测试集上评估 loss / accuracy，并输出精确率、召回率、F1、混淆矩阵；
    5. 保存模型到 models/emotion_cnn.h5；
    6. 保存训练曲线与混淆矩阵图片到 results/。

用法：
    python train.py
"""
import os

import matplotlib
matplotlib.use("Agg")  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

import config
from model import build_cnn_model


def load_dataset():
    """加载并归一化数据集。"""
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(
            f"未找到数据集 {config.DATASET_PATH}，请先运行: python prepare_dataset.py"
        )

    data = np.load(config.DATASET_PATH)
    x_train = data["x_train"].astype("float32") / 255.0
    y_train = data["y_train"].astype("int64")
    x_test = data["x_test"].astype("float32") / 255.0
    y_test = data["y_test"].astype("int64")

    print("x_train.shape, y_train.shape:", x_train.shape, y_train.shape)
    print("x_test.shape,  y_test.shape :", x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def plot_history(history, save_dir):
    """绘制并保存训练 / 验证的 loss 与 accuracy 曲线。"""
    # Loss 曲线
    plt.figure()
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # Accuracy 曲线
    plt.figure()
    plt.title("Training and Validation Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """绘制混淆矩阵热力图（仅依赖 matplotlib）。"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # 在每个格子中标注数量
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return cm


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    x_train, y_train, x_test, y_test = load_dataset()

    # 0. 打乱训练集：validation_split 取数组末尾 20%，不打乱会导致
    #    验证集只含末尾类别（FER2013 parquet 按标签排序），val_accuracy 恒为 0
    rng = np.random.default_rng(config.RANDOM_SEED)
    perm = rng.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    # 0.1 类别权重：FER2013 类别极不平衡（disgusted 仅 436 样本，happy 7215），
    #     不加权会让模型坍缩到只预测多数类
    class_weight = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(config.NUM_CLASSES),
        y=y_train,
    )
    class_weight = dict(enumerate(class_weight))
    print("类别权重:", class_weight)

    # 1. 构建模型
    model = build_cnn_model()
    model.summary()

    # 2. 训练（指导书: epochs=100, validation_split=0.2, batch_size=32，早停）
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=config.EARLY_STOP_PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=config.EPOCHS,
        validation_split=config.VALIDATION_SPLIT,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stop],
        class_weight=class_weight,
    )

    # 3. 测试集评估
    score = model.evaluate(x_test, y_test, batch_size=16)
    print("测试集 loss / accuracy:", score)

    # 4. 精确率、召回率、F1
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print("\n分类报告（精确率 / 召回率 / F1）:")
    print(classification_report(
        y_test, y_pred,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.EMOTION_LABELS,
        digits=4,
        zero_division=0,
    ))

    # 5. 混淆矩阵
    plot_confusion_matrix(
        y_test, y_pred, config.EMOTION_LABELS,
        os.path.join(config.RESULTS_DIR, "confusion_matrix.png"),
    )

    # 6. 训练曲线
    plot_history(history, config.RESULTS_DIR)

    # 7. 保存模型（与指导书一致使用 h5 格式）
    model.save(config.MODEL_PATH)
    print(f"模型已保存: {config.MODEL_PATH}")
    print(f"训练曲线与混淆矩阵已保存到: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
