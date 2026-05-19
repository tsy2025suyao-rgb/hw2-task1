import torch
import torch.nn as nn
from torchvision import models


def _get_resnet(model_name, pretrained):
    """加载 ResNet-18 或 ResNet-34，可选择是否使用 ImageNet 预训练权重。"""
    weights = None
    if pretrained:
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    if model_name == "resnet18":
        return models.resnet18(weights=weights)
    if model_name == "resnet34":
        return models.resnet34(weights=weights)
    raise ValueError(f"Unsupported model: {model_name}")


class SEBlock(nn.Module):
    """SE 注意力模块：学习每个通道的重要性权重。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class CBAMBlock(nn.Module):
    """CBAM 注意力模块：依次进行通道注意力和空间注意力。"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        padding = kernel_size // 2
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))
        channel_weights = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * channel_weights.view(b, c, 1, 1)

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        spatial_weights = self.spatial(torch.cat([avg_map, max_map], dim=1))
        return x * spatial_weights


def _attention_block(attention, channels):
    attention = attention.lower()
    if attention == "se":
        return SEBlock(channels)
    if attention == "cbam":
        return CBAMBlock(channels)
    raise ValueError(f"Unsupported attention type: {attention}")


def _parse_attention_layers(attention_layers):
    if attention_layers == "all":
        return ["layer1", "layer2", "layer3", "layer4"]
    return [name.strip() for name in attention_layers.split(",") if name.strip()]


def build_model(
    model_name="resnet18",
    num_classes=102,
    pretrained=True,
    attention="none",
    attention_layers="all",
):
    """
    统一模型构建入口。

    Baseline: attention="none"
    注意力实验: attention="se" 或 "cbam"
    attention_layers 可取 "all"、"layer4"、"layer3,layer4" 等。
    """
    model = _get_resnet(model_name, pretrained)

    if attention.lower() != "none":
        layer_channels = {
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
        }
        for layer_name in _parse_attention_layers(attention_layers):
            if layer_name not in layer_channels:
                raise ValueError(f"Unsupported attention layer: {layer_name}")
            original_layer = getattr(model, layer_name)
            block = _attention_block(attention, layer_channels[layer_name])
            setattr(model, layer_name, nn.Sequential(original_layer, block))

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
