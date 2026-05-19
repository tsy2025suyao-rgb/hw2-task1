import argparse
import csv
import math
import os

import torch
import torch.nn as nn

from dataset import get_dataloaders
from models import build_model
from utils import save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Flower102 训练脚本")

    parser.add_argument("--experiment", default="baseline_resnet18_pretrained")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--num-classes", type=int, default=102)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true", help="使用 ImageNet 预训练权重")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--attention", choices=["none", "se", "cbam"], default="none")
    parser.add_argument("--attention-layers", default="all")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=None, help="整个网络使用同一个学习率")
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-classifier", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)

    return parser.parse_args()


def build_optimizer(args, model):
    """创建优化器。传入 --lr 时关闭差分学习率。"""
    if args.lr is not None:
        print(f"Single learning rate: lr={args.lr}")
        if args.optimizer == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": args.lr_backbone, "name": "backbone"},
        {"params": classifier_params, "lr": args.lr_classifier, "name": "classifier"},
    ]
    print(
        "Differential learning rates: "
        f"backbone={args.lr_backbone}, classifier={args.lr_classifier}"
    )

    if args.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)


def build_scheduler(args, optimizer):
    """支持 warmup + cosine decay；warmup_epochs=0 时退化为普通 cosine。"""
    def lr_lambda(epoch):
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = epoch - args.warmup_epochs
        total = max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress / total))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_history(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join("checkpoints", f"{args.experiment}.pth")
    history_path = os.path.join("logs", f"{args.experiment}.csv")

    train_loader, val_loader, _ = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        attention=args.attention,
        attention_layers=args.attention_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_acc, args)
            print(f"保存最佳模型: {checkpoint_path}, val_acc={best_acc:.4f}")

    save_history(history, history_path)
    print(f"最佳验证集准确率: {best_acc:.4f}")
    print(f"训练日志保存到: {history_path}")


if __name__ == "__main__":
    main()
