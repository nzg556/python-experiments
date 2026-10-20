"""生成模拟水果病害数据集

由于无法直接下载真实数据集，这里用 Pillow 程序化生成「结构上有区分度」的
模拟图片：每种水果有特征底色，健康类目干净光滑，病害类目叠加斑点/纹理。
生成后可用真实数据直接替换 data/ 下对应目录即可。

用法:
    python generate_dataset.py

生成结构:
    data/
        train/<class_name>/*.jpg   (每类 120 张)
        val/<class_name>/*.jpg     (每类 30 张)
"""

import os
import random
import math

from PIL import Image, ImageDraw, ImageFilter

import config

# 各水果的底色 (R,G,B)
FRUIT_BASE_COLOR = {
    "Apple": (220, 60, 50),     # 红苹果
    "Grape": (90, 50, 140),    # 紫葡萄
    "Tomato": (210, 70, 40),   # 红番茄 (偏橙)
}

# 各病害的视觉特征
DISEASE_STYLE = {
    "Healthy":     {"spots": 0,  "spot_color": None,        "darken": 0.0,  "texture": "smooth"},
    "Scab":         {"spots": 25, "spot_color": (80, 50, 30),  "darken": 0.15, "texture": "rough"},
    "Rust":         {"spots": 20, "spot_color": (200, 120, 40),"darken": 0.05, "texture": "smooth"},
    "BlackRot":     {"spots": 18, "spot_color": (30, 25, 20),  "darken": 0.25, "texture": "rough"},
    "EarlyBlight":  {"spots": 22, "spot_color": (120, 70, 30), "darken": 0.10, "texture": "ring"},
    "LateBlight":   {"spots": 16, "spot_color": (50, 60, 40),  "darken": 0.20, "texture": "blotchy"},
}


def _blend(color, factor, target=0):
    """颜色向黑(0)或白(255)插值。factor>0 变暗, <0 变亮。"""
    if factor >= 0:
        return tuple(int(c * (1 - factor)) for c in color)
    else:
        f = -factor
        return tuple(int(c + (255 - c) * f) for c in color)


def _add_gaussian_noise(img, intensity=8):
    """给图片叠加高斯噪声，增加样本多样性。"""
    import numpy as np
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.normal(0, intensity, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_image(fruit, disease, size=224, rng=None):
    """生成一张模拟水果病害图片。

    Args:
        fruit: 水果类型 'Apple'/'Grape'/'Tomato'
        disease: 病害类型 (见 DISEASE_STYLE 的键)
        size: 图片边长
        rng: random.Random 实例
    """
    if rng is None:
        rng = random.Random()

    style = DISEASE_STYLE[disease]
    base = FRUIT_BASE_COLOR[fruit]

    # 背景：浅色，略带变化
    bg_color = (245, 240, 230)
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    # 画一个水果主体（椭圆/圆），位置略随机
    cx = size // 2 + rng.randint(-10, 10)
    cy = size // 2 + rng.randint(-10, 10)
    rx = rng.randint(size // 4, size // 3)
    ry = rng.randint(int(rx * 0.9), int(rx * 1.1))

    # 主体颜色带底色变化
    fruit_color = tuple(
        max(0, min(255, c + rng.randint(-25, 25))) for c in base
    )
    fruit_color = _blend(fruit_color, style["darken"])

    # 画水果主体
    draw.ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry],
        fill=fruit_color,
    )

    # 高光（让水果更立体）
    hl_color = _blend(fruit_color, -0.35)
    hl_rx = int(rx * 0.35)
    hl_ry = int(ry * 0.35)
    draw.ellipse(
        [cx - hl_rx - int(rx * 0.2), cy - hl_ry - int(ry * 0.2),
         cx - hl_rx + int(rx * 0.1), cy - hl_ry + int(ry * 0.1)],
        fill=hl_color,
    )

    # 病害斑点
    n_spots = style["spots"]
    spot_color = style["spot_color"]
    if n_spots > 0 and spot_color is not None:
        for _ in range(n_spots):
            # 在水果区域内随机放斑点
            angle = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0, min(rx, ry) * 0.8)
            sx = int(cx + math.cos(angle) * r)
            sy = int(cy + math.sin(angle) * r)
            spot_r = rng.randint(3, 10)

            if style["texture"] == "ring":
                # 同心环状斑点（早疫病特征）
                draw.ellipse([sx - spot_r, sy - spot_r,
                              sx + spot_r, sy + spot_r],
                             outline=spot_color, width=2)
                inner = int(spot_r * 0.5)
                draw.ellipse([sx - inner, sy - inner,
                              sx + inner, sy + inner],
                             fill=spot_color)
            elif style["texture"] == "blotchy":
                # 不规则暗斑块（晚疫病特征）
                for _ in range(3):
                    bx = sx + rng.randint(-8, 8)
                    by = sy + rng.randint(-8, 8)
                    br = rng.randint(4, 9)
                    col = tuple(max(0, c - rng.randint(0, 20)) for c in spot_color)
                    draw.ellipse([bx - br, by - br, bx + br, by + br], fill=col)
            else:
                # 普通圆形斑点
                col = spot_color
                if style["texture"] == "rough":
                    col = tuple(max(0, c - rng.randint(0, 15)) for c in spot_color)
                draw.ellipse([sx - spot_r, sy - spot_r,
                              sx + spot_r, sy + spot_r], fill=col)

    # 轻微模糊让边缘更自然
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    # 加噪声
    img = _add_gaussian_noise(img, intensity=rng.randint(5, 12))

    # 随机水平翻转
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img


def _parse_class(class_name):
    """从类名拆出 fruit 和 disease。如 'Apple_Scab' -> ('Apple', 'Scab')"""
    parts = class_name.split("_", 1)
    return parts[0], parts[1]


def main():
    rng = random.Random(config.RANDOM_SEED)
    n_train = 120
    n_val = 30

    total = 0
    for class_name in config.CLASS_NAMES:
        fruit, disease = _parse_class(class_name)
        if fruit not in FRUIT_BASE_COLOR:
            print(f"[skip] 未知水果: {fruit} (类目 {class_name})")
            continue
        if disease not in DISEASE_STYLE:
            print(f"[skip] 未知病害: {disease} (类目 {class_name})")
            continue

        for split, n, base_dir in [
            ("train", n_train, config.TRAIN_DIR),
            ("val", n_val, config.VAL_DIR),
        ]:
            out_dir = os.path.join(base_dir, class_name)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(n):
                img = generate_image(fruit, disease, size=config.IMAGE_SIZE, rng=rng)
                fname = f"{class_name}_{split}_{i:03d}.jpg"
                img.save(os.path.join(out_dir, fname), quality=90)
                total += 1
        print(f"[ok] {class_name}: {n_train} train + {n_val} val")

    # 预测目录放几张未标注图供 predict.py 演示
    os.makedirs(config.PREDICT_DIR, exist_ok=True)
    rng2 = random.Random(0)
    for i in range(8):
        cn = config.CLASS_NAMES[rng2.randrange(config.NUM_CLASSES)]
        fruit, disease = _parse_class(cn)
        img = generate_image(fruit, disease, size=config.IMAGE_SIZE, rng=rng2)
        img.save(os.path.join(config.PREDICT_DIR, f"sample_{i}.jpg"), quality=90)
    print(f"[ok] 预测样本: 8 张 -> {config.PREDICT_DIR}")

    print(f"\n数据集生成完成，共 {total} 张图片。")
    print(f"  训练集: {config.TRAIN_DIR}")
    print(f"  验证集: {config.VAL_DIR}")
    print(f"\n提示: 如需使用真实数据，将 data/train 和 data/val 下各子目录替换为真实图片即可。")


if __name__ == "__main__":
    main()
