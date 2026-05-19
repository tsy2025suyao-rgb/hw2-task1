import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    """固定随机种子，方便不同实验之间做公平对比。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_checkpoint(path, model, optimizer, epoch, best_acc, args):
    """保存验证集准确率最高的模型。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "args": vars(args),
        },
        path,
    )
