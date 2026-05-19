from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ImageNet 预训练模型常用的 RGB 均值和标准差。
# 使用预训练 ResNet 时，输入归一化方式最好与预训练阶段保持一致。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size=224):
    """构建训练集和验证/测试集的数据预处理。"""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def get_dataloaders(data_root="./data", batch_size=32, image_size=224, num_workers=0):
    """读取 Flower102 官方 train/val/test 划分，并返回三个 DataLoader。"""
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = datasets.Flowers102(
        root=data_root,
        split="train",
        transform=train_transform,
        download=False,
    )
    val_dataset = datasets.Flowers102(
        root=data_root,
        split="val",
        transform=eval_transform,
        download=False,
    )
    test_dataset = datasets.Flowers102(
        root=data_root,
        split="test",
        transform=eval_transform,
        download=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
