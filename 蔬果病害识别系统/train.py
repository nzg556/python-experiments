"""训练主脚本

流程:
    1. 构造数据加载器与 ResNet 模型
    2. 交叉熵损失 + SGD (带动量与权重衰减)
    3. StepLR 学习率衰减
    4. 每 epoch 评估验证集，保存最佳/最新检查点
    5. 训练结束绘制曲线

用法:
    python train.py                 # 从头训练
    python train.py --resume        # 从最新检查点继续
"""

import os
import argparse
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import build_model, get_device, count_parameters
from utils import set_seed, save_checkpoint, AverageMeter, compute_accuracy, plot_training_curves, load_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """训练一个 epoch。"""
    model.train()
    loss_meter = AverageMeter()
    correct, total = 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        c, t = compute_accuracy(logits, labels)
        correct += c
        total += t
        loss_meter.update(loss.item(), n=images.size(0))

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{correct / total:.4f}")
    pbar.close()

    return loss_meter.avg, correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """在验证集上评估。"""
    model.eval()
    loss_meter = AverageMeter()
    correct, total = 0, 0

    pbar = tqdm(loader, desc="Validate", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        c, t = compute_accuracy(logits, labels)
        correct += c
        total += t
        loss_meter.update(loss.item(), n=images.size(0))
    pbar.close()

    return loss_meter.avg, correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="水果病害识别 - ResNet 训练")
    parser.add_argument("--resume", action="store_true", help="从最新检查点继续训练")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认 epoch 数")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print(f"设备: {device}")
    print(f"模型: {config.MODEL_NAME}, 预训练: {config.PRETRAINED}")

    # 数据
    train_loader, val_loader = get_dataloaders()

    # 模型
    model = build_model(device=device)
    total_p, train_p = count_parameters(model)
    print(f"总参数: {total_p:,}, 可训练: {train_p:,}")

    # 损失 / 优化器 / 调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

    num_epochs = config.NUM_EPOCHS if args.epochs is None else args.epochs
    start_epoch = 0
    best_val_acc = 0.0

    # 续训
    if args.resume and os.path.isfile(config.LAST_MODEL_PATH):
        ckpt = load_checkpoint(config.LAST_MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"从 epoch {start_epoch} 继续训练，历史最佳验证准确率: {best_val_acc:.4f}")

    # 训练记录
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    print(f"\n开始训练，共 {num_epochs} 个 epoch (起始 epoch={start_epoch})\n")
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={lr_now:.2e} | {dt:.1f}s"
        )

        # 保存最新
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
            },
            config.LAST_MODEL_PATH,
        )

        # 保存最佳
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                config.BEST_MODEL_PATH,
            )
            print(f"  -> 新的最佳模型已保存 (val_acc={best_val_acc:.4f})")

    # 训练曲线
    plot_training_curves(history)

    print(f"\n训练完成。最佳验证准确率: {best_val_acc:.4f}")
    print(f"最佳模型: {config.BEST_MODEL_PATH}")
    print(f"最新模型: {config.LAST_MODEL_PATH}")


if __name__ == "__main__":
    main()
