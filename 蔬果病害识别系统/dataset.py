"""数据集与数据加载

- FruitDiseaseDataset: 按 ImageFolder 方式从目录读取图片
- get_transforms: 训练集做数据增强，验证集只做 resize+normalize
- get_dataloaders: 构造训练/验证 DataLoader
"""

import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


class FruitDiseaseDataset(Dataset):
    """从目录结构 data/<split>/<class_name>/*.jpg 加载数据集。

    Args:
        root: data/train 或 data/val
        transform: torchvision transforms 管线
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []  # [(图片路径, 类别索引), ...]

        if not os.path.isdir(root):
            raise FileNotFoundError(f"数据目录不存在: {root}\n请先运行 generate_dataset.py")

        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            idx = config.CLASS_TO_IDX[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append((os.path.join(class_dir, fname), idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"目录 {root} 下没有找到任何图片。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_transforms():
    """返回 (train_transform, val_transform)。"""
    train_transform = transforms.Compose([
        transforms.Resize((config.TRAIN_RESIZE, config.TRAIN_RESIZE)),
        transforms.RandomResizedCrop(
            config.IMAGE_SIZE,
            scale=(0.7, 1.0),
            ratio=(0.8, 1.2),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.VAL_RESIZE, config.VAL_RESIZE)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])
    return train_transform, val_transform


def get_dataloaders():
    """构造训练集与验证集的 DataLoader。"""
    train_t, val_t = get_transforms()

    train_dataset = FruitDiseaseDataset(config.TRAIN_DIR, transform=train_t)
    val_dataset = FruitDiseaseDataset(config.VAL_DIR, transform=val_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
    )

    print(f"训练集: {len(train_dataset)} 张图片, {len(train_loader)} 个 batch")
    print(f"验证集: {len(val_dataset)} 张图片, {len(val_loader)} 个 batch")
    return train_loader, val_loader
