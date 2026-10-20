# -*- coding: utf-8 -*-
"""
全局配置文件
================
《神经网络与深度学习实训》——基于 CNN 的人脸表情（情绪）识别项目

数据集目录结构（与实训指导书一致）::

    emotion_recognition/
        train/                  <- 自行准备的图片数据
            angry/              0 生气
                xxx.jpg
            disgusted/          1 厌恶
            fearful/            2 恐惧
            happy/              3 开心
            neutral/            4 平静
            sad/                5 伤心
            surprised/          6 惊讶
"""
import os

# ---------------------------------------------------------------- 路径配置
# 项目根目录（本文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始图片目录：每个子文件夹代表一种表情
TRAIN_DIR = os.path.join(BASE_DIR, "train")

# 原始真实数据集存放目录（如下载的 FER2013 parquet / csv）
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
HF_FER2013_DIR = os.path.join(DATASETS_DIR, "fer2013_hf")

# 预处理后生成的数据集文件
DATASET_PATH = os.path.join(BASE_DIR, "data.npz")

# 模型保存目录与模型文件
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_cnn.h5")

# 训练曲线、混淆矩阵等结果输出目录
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------- 数据配置
# 表情标签，索引即标签编号（顺序与指导书一致，且恰好为字母序）
EMOTION_LABELS = ["angry", "disgusted", "fearful", "happy",
                  "neutral", "sad", "surprised"]
# 中文标签，用于界面与摄像头画面显示
EMOTION_LABELS_ZH = ["生气", "厌恶", "恐惧", "开心",
                     "平静", "伤心", "惊讶"]
NUM_CLASSES = len(EMOTION_LABELS)

# 指导书统一使用 100x100 灰度图
IMG_SIZE = 100
CHANNELS = 1

# 测试集比例 20%
TEST_RATIO = 0.2
RANDOM_SEED = 42

# 支持的图片后缀
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------------------------------------------------------- 训练配置
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-2  # SGD 默认 1e-3 收敛太慢，提到 1e-2
VALIDATION_SPLIT = 0.2
EARLY_STOP_PATIENCE = 3

# ---------------------------------------------------------------- 检测配置
# 人脸检测置信度阈值
FACE_CONF_THRESHOLD = 0.7

# 中文字体候选（Windows 下微软雅黑 / 黑体）
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
