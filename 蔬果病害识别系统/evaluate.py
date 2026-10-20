"""模型评估脚本

加载最佳检查点，在验证集上计算：准确率、精确率、召回率、F1、混淆矩阵。

用法:
    python evaluate.py
"""

import os

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import get_dataloaders, get_transforms, FruitDiseaseDataset
from model import build_model, get_device
from utils import load_checkpoint, plot_confusion_matrix


@torch.no_grad()
def evaluate_model(model, loader, device):
    """返回 (all_labels, all_preds, all_probs)。"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return all_labels, all_preds, all_probs


def print_classification_report(labels, preds):
    """打印分类报告。"""
    from sklearn.metrics import classification_report
    report = classification_report(
        labels, preds, target_names=config.CLASS_NAMES, digits=4
    )
    print("\n分类报告:\n")
    print(report)

    # 保存到文件
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"分类报告已保存: {report_path}")


def main():
    device = get_device()
    print(f"设备: {device}")

    if not os.path.isfile(config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"找不到最佳模型: {config.BEST_MODEL_PATH}\n请先运行 python train.py"
        )

    # 数据
    _, val_t = get_transforms()
    val_dataset = FruitDiseaseDataset(config.VAL_DIR, transform=val_t)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    print(f"验证集: {len(val_dataset)} 张图片")

    # 模型
    model = build_model(device=device, pretrained=False)
    ckpt = load_checkpoint(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"加载检查点: {config.BEST_MODEL_PATH} (epoch={ckpt.get('epoch')})")

    # 评估
    labels, preds, probs = evaluate_model(model, val_loader, device)

    # 报告 + 混淆矩阵
    print_classification_report(labels, preds)
    plot_confusion_matrix(labels, preds)

    # 总体准确率
    correct = sum(int(p == l) for p, l in zip(preds, labels))
    print(f"\n总体准确率: {correct}/{len(labels)} = {correct / len(labels):.4f}")


if __name__ == "__main__":
    main()
