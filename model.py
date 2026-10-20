# -*- coding: utf-8 -*-
"""
CNN 模型定义
================
对应实训指导书“（2）模型训练”中的网络结构：

    Conv2D(32, 5x5) + MaxPool
    Conv2D(64, 3x3) + MaxPool
    Conv2D(64, 3x3)
    Flatten -> Dense(512) + Dropout(0.5)
             -> Dense(128) + Dropout(0.2)
             -> Dense(7, softmax)
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)

import config


def build_cnn_model():
    """构建并编译用于 7 类表情分类的 CNN 模型。"""
    model = Sequential([
        # 第一层卷积：32 个 5x5 卷积核，same 填充，输入为 100x100x1 灰度图
        Conv2D(32, kernel_size=(5, 5), padding="same",
               activation=tf.nn.relu,
               input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS)),
        MaxPooling2D(pool_size=(2, 2), strides=2),

        # 第二层卷积：64 个 3x3 卷积核
        Conv2D(64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2, 2), strides=2),

        # 第三层卷积：继续提取更抽象的特征
        Conv2D(64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu),

        # 全连接分类头
        Flatten(),
        Dense(512, activation="relu"),   # 参数量较大，用 Dropout 防止过拟合
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(config.NUM_CLASSES, activation="softmax"),
    ])

    # 标签为整数（非 one-hot），使用 SparseCategoricalCrossentropy；
    # 末层已有 softmax，故 from_logits=False
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
