# -*- coding: utf-8 -*-
"""
模块（1）：将真实图片制作成数据集
====================================
对应实训指导书“（1）将图片制作成数据集”。

支持三种真实数据来源：
    1. 【推荐】Hugging Face 上的 FER2013 官方镜像（parquet，共 35887 张真实人脸，
       48x48 灰度，保留官方 28709 训练 / 7178 测试划分）：
           python prepare_dataset.py --hf
       数据来源（可追溯）：
           https://hf-mirror.com/datasets/Jeneral/fer-2013
           （Kaggle 官方 FER2013: msambare/fer2013 的镜像）

    2. 本地 fer2013.csv（Kaggle FER2013，列: emotion,pixels,Usage）。
       优先按官方 Usage 列划分（Training / PublicTest / PrivateTest）：
           python prepare_dataset.py --csv fer2013.csv

    3. 自己采集的图片，按表情分文件夹放在 train/ 下（8:2 随机划分）：
           python prepare_dataset.py --data_dir train

输出：
    data.npz  —— x_train/y_train/x_test/y_test，图片统一缩放为 100x100x1 灰度。
"""
import argparse
import io
import os
import random
import time
import urllib.request

import cv2
import numpy as np

import config

# ---------------------------------------------------------------- 数据源
# Hugging Face 镜像（hf-mirror.com 为 Hugging Face 官方国内镜像），
# 读取的是数据集的官方 parquet 自动转换文件，与原始 .pt/.csv 内容一致。
_HF_BASE = ("https://hf-mirror.com/datasets/Jeneral/fer-2013/resolve/"
            "refs%2Fconvert%2Fparquet")
_HF_FILES = {
    "train-0000.parquet": _HF_BASE + "/default/train/0000.parquet",
    "test-0000.parquet": _HF_BASE + "/default/test/0000.parquet",
}

# FER2013(fer2013.csv 官方编号) -> 本项目标签编号（本项目顺序与指导书一致）
# csv 官方: 0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral
# 本项目:   0 angry, 1 disgusted, 2 fearful, 3 happy,
#           4 neutral, 5 sad, 6 surprised
_FER2013_CSV_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 4}

# HF parquet 的标签编号（其 ClassLabel 顺序）与本项目完全一致：
# ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ---------------------------------------------------------------- 通用工具
def _resize_gray(image):
    """将灰度图统一缩放到 IMG_SIZE x IMG_SIZE，并保留 (H, W, 1) 通道维。"""
    image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE),
                       interpolation=cv2.INTER_AREA)
    return image[..., np.newaxis]


def train_test_split(images, labels, test_ratio, seed):
    """打乱样本并按比例划分训练集 / 测试集。"""
    rng = random.Random(seed)
    indices = list(range(len(images)))
    rng.shuffle(indices)

    test_count = int(len(images) * test_ratio)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return (images[train_idx], labels[train_idx],
            images[test_idx], labels[test_idx])


def print_split_stats(y_train, y_test):
    """打印训练/测试集各类别样本数。"""
    print("各类别样本数:")
    for i, name in enumerate(config.EMOTION_LABELS):
        n_tr = int(np.sum(y_train == i))
        n_te = int(np.sum(y_test == i))
        print(f"  {i} {name:10s}: train={n_tr:5d}  test={n_te:4d}")


# ---------------------------------------------------------------- 来源1：图片文件夹
def load_from_image_folders(data_dir):
    """
    读取按类别分文件夹存放的图片数据，返回 images / labels（8:2 划分在 main 中完成）。

    返回:
        images: uint8, shape=(N, 100, 100, 1)
        labels: uint8, shape=(N,)
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir}\n"
            f"请先在项目根目录下创建 train/ 目录，并按类别建立 7 个子文件夹：\n"
            f"{', '.join(config.EMOTION_LABELS)}"
        )

    # 子文件夹名（标签名），排序以保证标签编号固定
    label_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    if not label_names:
        raise ValueError(f"目录 {data_dir} 下没有找到任何类别子文件夹。")

    label_to_index = {name: i for i, name in enumerate(label_names)}
    print("标签映射:", label_to_index)

    if set(label_names) != set(config.EMOTION_LABELS):
        print("[警告] 子文件夹名称与 config.EMOTION_LABELS 不完全一致，"
              "请确认标签顺序与训练/检测时使用的顺序相同！")

    images, labels = [], []
    for label_name in label_names:
        class_dir = os.path.join(data_dir, label_name)
        file_names = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(config.IMAGE_EXTS)]
        print(f"  {label_name}: {len(file_names)} 张")
        for file_name in file_names:
            image = cv2.imread(os.path.join(class_dir, file_name),
                               cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"    [跳过] 无法读取: {file_name}")
                continue
            images.append(_resize_gray(image))
            labels.append(label_to_index[label_name])

    return np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.uint8)


# ---------------------------------------------------------------- 来源2：fer2013.csv
def load_from_fer_csv(csv_path):
    """
    读取 fer2013.csv（列: emotion,pixels,Usage）。

    若存在 Usage 列，按 FER2013 官方划分：
        Training -> 训练集；PublicTest + PrivateTest -> 测试集；
    否则回退为 8:2 随机划分。
    直接返回 4 个数组 (x_train, y_train, x_test, y_test)。
    """
    import csv

    train_imgs, train_lbs, test_imgs, test_lbs = [], [], [], []
    has_usage = None
    total = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
            if pixels.size != 48 * 48:
                continue
            image = pixels.reshape(48, 48)
            image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE),
                               interpolation=cv2.INTER_CUBIC)
            label = _FER2013_CSV_LABEL_MAP[int(row["emotion"])]
            total += 1

            if has_usage is None:
                has_usage = "Usage" in row and row["Usage"]
            if has_usage:
                if row["Usage"] == "Training":
                    train_imgs.append(image)
                    train_lbs.append(label)
                else:  # PublicTest / PrivateTest
                    test_imgs.append(image)
                    test_lbs.append(label)
            else:
                train_imgs.append(image)
                train_lbs.append(label)

    print(f"CSV 共读取 {total} 条，官方 Usage 划分: {bool(has_usage)}")

    train_imgs = np.asarray(train_imgs, dtype=np.uint8)[..., np.newaxis]
    train_lbs = np.asarray(train_lbs, dtype=np.uint8)
    test_imgs = np.asarray(test_imgs, dtype=np.uint8)[..., np.newaxis]
    test_lbs = np.asarray(test_lbs, dtype=np.uint8)

    if not has_usage:
        return train_test_split(train_imgs[..., 0], train_lbs,
                                config.TEST_RATIO, config.RANDOM_SEED)
    return train_imgs, train_lbs, test_imgs, test_lbs


# ---------------------------------------------------------------- 来源3：HF parquet
def download_hf_fer2013(dest_dir):
    """
    从 hf-mirror.com 下载 FER2013 官方 parquet 文件（约 53MB）。
    已存在的文件会跳过，支持断点重复执行。
    """
    os.makedirs(dest_dir, exist_ok=True)
    for name, url in _HF_FILES.items():
        dest = os.path.join(dest_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"已存在 {name} ({os.path.getsize(dest)/1e6:.1f} MB)，跳过")
            continue

        print(f"下载 {name} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done, last = 0, time.time()
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 3:
                    pct = f"{done/total*100:5.1f}%" if total else ""
                    print(f"    {done/1e6:6.1f} MB {pct}")
                    last = time.time()
        print(f"    完成: {os.path.getsize(dest)/1e6:.1f} MB")


def _decode_parquet_split(parquet_path):
    """读取单个 parquet 文件并解码为 (images, labels)。"""
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(parquet_path)
    if "img_bytes" not in df.columns or "labels" not in df.columns:
        raise ValueError(f"{parquet_path} 缺少 img_bytes/labels 列，"
                         f"实际列为 {list(df.columns)}")

    images, labels = [], []
    for img_bytes, label in zip(df["img_bytes"], df["labels"]):
        # datasets-server 的图像字段可能是 dict({'bytes': ...}) 或裸 bytes
        if isinstance(img_bytes, dict):
            img_bytes = img_bytes["bytes"]
        image = np.array(Image.open(io.BytesIO(img_bytes)).convert("L"),
                         dtype=np.uint8)
        images.append(_resize_gray(image))
        labels.append(int(label))

    return np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.uint8)


def load_from_hf_parquet(train_path, test_path):
    """读取下载好的 train/test parquet，直接返回官方划分的 4 个数组。"""
    print("解析训练集 parquet ...")
    x_train, y_train = _decode_parquet_split(train_path)
    print("解析测试集 parquet ...")
    x_test, y_test = _decode_parquet_split(test_path)
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description="制作真实表情识别数据集 data.npz")
    parser.add_argument("--data_dir", default=config.TRAIN_DIR,
                        help="图片分类目录（默认 ./train），无其他参数时使用")
    parser.add_argument("--csv", default=None,
                        help="fer2013.csv 路径（按官方 Usage 划分）")
    parser.add_argument("--hf", action="store_true",
                        help="从 HF 镜像下载 FER2013 parquet 并制作数据集")
    parser.add_argument("--hf_dir", default=config.HF_FER2013_DIR,
                        help="HF parquet 存放目录（默认 ./datasets/fer2013_hf）")
    parser.add_argument("--parquet_train", default=None,
                        help="直接指定本地训练集 parquet 文件")
    parser.add_argument("--parquet_test", default=None,
                        help="直接指定本地测试集 parquet 文件")
    parser.add_argument("--output", default=config.DATASET_PATH,
                        help="输出 npz 文件路径（默认 ./data.npz）")
    args = parser.parse_args()

    official_split = False  # 是否使用数据集官方划分

    if args.parquet_train or args.parquet_test:
        if not (args.parquet_train and args.parquet_test):
            raise ValueError("--parquet_train 与 --parquet_test 必须同时提供")
        print(f"从本地 parquet 读取: {args.parquet_train}, {args.parquet_test}")
        x_train, y_train, x_test, y_test = load_from_hf_parquet(
            args.parquet_train, args.parquet_test)
        official_split = True
    elif args.hf:
        download_hf_fer2013(args.hf_dir)
        x_train, y_train, x_test, y_test = load_from_hf_parquet(
            os.path.join(args.hf_dir, "train-0000.parquet"),
            os.path.join(args.hf_dir, "test-0000.parquet"))
        official_split = True
    elif args.csv:
        print(f"从 CSV 读取数据集: {args.csv}")
        x_train, y_train, x_test, y_test = load_from_fer_csv(args.csv)
        official_split = True
    else:
        print(f"从图片文件夹读取数据集: {args.data_dir}")
        images, labels = load_from_image_folders(args.data_dir)
        if len(images) == 0:
            raise RuntimeError("没有读取到任何图片，请检查数据目录。")
        x_train, y_train, x_test, y_test = train_test_split(
            images, labels, config.TEST_RATIO, config.RANDOM_SEED)

    print(f"\n图片总数: {len(x_train) + len(x_test)}")
    print(f"训练集: x={x_train.shape}, y={y_train.shape}")
    print(f"测试集: x={x_test.shape}, y={y_test.shape}")
    print_split_stats(y_train, y_test)
    if official_split:
        print("（使用数据集官方 train/test 划分）")

    # 像素以 uint8(0~255) 保存以压缩体积；训练 / 预测时再除以 255 归一化
    np.savez_compressed(
        args.output,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    print(f"真实数据集已保存: {args.output}")


if __name__ == "__main__":
    main()
