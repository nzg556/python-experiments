"""全局配置：水果病害识别 (ResNet)

所有路径、超参数、类别定义集中管理，方便统一调参。
"""

import os

# ---------------- 项目根目录 ----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------- 数据路径 ----------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
PREDICT_DIR = os.path.join(DATA_DIR, "predict")  # 待预测图片放这里

# ---------------- 输出路径 ----------------
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- 类别定义 ----------------
# 8 个类别：覆盖苹果、葡萄、番茄的健康与常见病害
CLASS_NAMES = [
    "Apple_Healthy",
    "Apple_Scab",
    "Apple_Rust",
    "Grape_Healthy",
    "Grape_BlackRot",
    "Tomato_Healthy",
    "Tomato_EarlyBlight",
    "Tomato_LateBlight",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ---------------- 模型 ----------------
IMAGE_SIZE = 224          # ResNet 标准输入
MODEL_NAME = "resnet18"   # 可选: resnet18 / resnet34 / resnet50
PRETRAINED = True         # 使用 ImageNet 预训练权重 (迁移学习)

# ---------------- 训练超参数 ----------------
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# 学习率调度
LR_STEP_SIZE = 5      # 每 5 个 epoch 衰减
LR_GAMMA = 0.5        # 衰减系数

# ---------------- 数据增强 ----------------
# 训练集做较强增强，验证集仅 resize + normalize
TRAIN_RESIZE = 256
VAL_RESIZE = 256

# ---------------- 其他 ----------------
NUM_WORKERS = 0        # Windows 下建议 0，避免多进程开销
RANDOM_SEED = 42
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")

# ImageNet 标准化常量 (与预训练权重一致)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
