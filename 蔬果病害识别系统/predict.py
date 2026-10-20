"""推理脚本：对单张图片或目录下所有图片做预测。

用法:
    python predict.py --image path/to/image.jpg
    python predict.py --dir data/predict
    python predict.py --image path/to/image.jpg --topk 3
"""

import os
import argparse

import torch
from PIL import Image
from torchvision import transforms

import config
from model import build_model, get_device
from utils import load_checkpoint


def preprocess_image(image_path, transform):
    """读取并预处理单张图片，返回 [1, C, H, W] tensor。"""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # 加 batch 维
    return tensor


@torch.no_grad()
def predict(model, image_path, transform, device, topk=1):
    """对单张图片预测，返回 [(class_name, prob), ...]。"""
    model.eval()
    tensor = preprocess_image(image_path, transform).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]

    topk = min(topk, len(config.CLASS_NAMES))
    top_probs, top_indices = torch.topk(probs, k=topk)
    results = [
        (config.IDX_TO_CLASS[int(idx)], float(p))
        for p, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())
    ]
    return results


def visualize_prediction(image_path, results):
    """可视化预测结果并保存。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})

    ax1.imshow(img)
    ax1.set_title("Input")
    ax1.axis("off")

    names = [r[0] for r in results]
    probs = [r[1] for r in results]
    colors = ["#2ecc71" if i == 0 else "#95a5a6" for i in range(len(names))]
    ax2.barh(range(len(names)), probs, color=colors)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Probability")
    ax2.invert_yaxis()
    ax2.set_title("Prediction")

    plt.tight_layout()
    save_path = os.path.join(
        config.RESULTS_DIR,
        f"pred_{os.path.splitext(os.path.basename(image_path))[0]}.png",
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description="水果病害识别 - 推理")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--dir", type=str, help="目录路径 (预测目录下所有图片)")
    parser.add_argument("--topk", type=int, default=3, help="输出 top-k 结果")
    args = parser.parse_args()

    if not args.image and not args.dir:
        args.dir = config.PREDICT_DIR  # 默认用 data/predict

    device = get_device()
    print(f"设备: {device}")

    if not os.path.isfile(config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"找不到模型: {config.BEST_MODEL_PATH}\n请先运行 python train.py"
        )

    # 推理用验证 transform
    _, val_transform = get_transforms_for_predict()

    model = build_model(device=device, pretrained=False)
    ckpt = load_checkpoint(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"加载模型: {config.BEST_MODEL_PATH}\n")

    # 收集待预测图片
    image_paths = []
    if args.image:
        image_paths = [args.image]
    else:
        d = args.dir
        if not os.path.isdir(d):
            print(f"目录不存在: {d}")
            return
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(d, fname))

    if not image_paths:
        print("没有找到可预测的图片。")
        return

    print(f"共 {len(image_paths)} 张图片\n")
    for path in image_paths:
        results = predict(model, path, val_transform, device, topk=args.topk)
        print(f"图片: {path}")
        for name, prob in results:
            print(f"  {name:25s} {prob*100:6.2f}%")
        viz = visualize_prediction(path, results)
        print(f"  可视化: {viz}\n")


def get_transforms_for_predict():
    """推理用的 transform (同验证集)。"""
    from dataset import get_transforms
    _, val_t = get_transforms()
    return None, val_t


if __name__ == "__main__":
    main()
