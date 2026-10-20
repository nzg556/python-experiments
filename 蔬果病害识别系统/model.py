"""ResNet 模型定义 (迁移学习)

支持 resnet18 / resnet34 / resnet50，加载 ImageNet 预训练权重，
替换最后的全连接层以适配水果病害分类的类别数。
"""

import torch
import torch.nn as nn
from torchvision import models

import config


def build_model(num_classes=None, pretrained=None, model_name=None, device=None):
    """构造 ResNet 分类模型。

    Args:
        num_classes: 类别数，默认取 config.NUM_CLASSES
        pretrained: 是否使用预训练权重，默认取 config.PRETRAINED
        model_name: 模型名 resnet18/34/50，默认取 config.MODEL_NAME
        device: torch.device

    Returns:
        nn.Module (已 .to(device) 或交由调用方转移)
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if pretrained is None:
        pretrained = config.PRETRAINED
    if model_name is None:
        model_name = config.MODEL_NAME

    # torchvision 新版本用 weights 参数，老版本用 pretrained
    weights_arg = _resolve_weights(model_name, pretrained)

    if model_name == "resnet18":
        model = models.resnet18(weights=weights_arg)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights_arg)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights_arg)
    else:
        raise ValueError(f"不支持的模型: {model_name}，可选 resnet18/34/50")

    # 替换分类头
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 迁移学习策略：冻结除最后一层外的所有参数
    # （注释掉下面这段即可改为全网络微调）
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    if device is not None:
        model = model.to(device)

    return model


def _resolve_weights(model_name, pretrained):
    """兼容 torchvision 新旧版本的权重参数。"""
    if not pretrained:
        return None
    try:
        # 新版 API
        from torchvision.models import get_model_weights
        weights_enum = get_model_weights(model_name)
        return weights_enum.DEFAULT
    except Exception:
        # 老版 API: 直接传 bool
        return True


def count_parameters(model):
    """统计可训练参数数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_device():
    """自动选择设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"使用设备: {device}")
    model = build_model(device=device)
    total, trainable = count_parameters(model)
    print(f"模型: {config.MODEL_NAME}")
    print(f"总参数量: {total:,}")
    print(f"可训练参数: {trainable:,}")
    print(f"类别数: {config.NUM_CLASSES}")
