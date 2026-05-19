import argparse

import torch
import torch.nn as nn

from dataset import get_dataloaders
from models import build_model
from train import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Flower102 测试脚本")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--num-classes", type=int, default=102)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--attention", choices=["none", "se", "cbam"], default="none")
    parser.add_argument("--attention-layers", default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    # 测试时权重来自 checkpoint，不需要重新下载 ImageNet 预训练权重。
    model = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=False,
        attention=args.attention,
        attention_layers=args.attention_layers,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
