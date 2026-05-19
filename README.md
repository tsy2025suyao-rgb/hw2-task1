# Flower102 迁移学习实验说明

本项目完成 102 Category Flower Dataset 上的图像分类实验，包括：

1. ImageNet 预训练 ResNet-18 Baseline 微调；
2. 超参数分析；
3. 预训练消融实验；
4. SEBlock / CBAM 注意力机制对比实验。

## 1. 工程结构

```text
hw2/
├── dataset.py              # Flower102 数据集读取与数据增强
├── models.py               # ResNet-18/34、SEBlock、CBAMBlock
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── utils.py                # 随机种子、checkpoint 保存等工具函数
├── requirements.txt        # Python 依赖
├── data/                   # Flower102 数据
├── checkpoints/            # 训练得到的模型权重
├── logs/                   # 训练日志 CSV 和曲线图
└── report/                 # 最终实验报告与图表
```

核心代码只有以下几个文件：

- `dataset.py`
- `models.py`
- `train.py`
- `test.py`
- `utils.py`

## 2. 环境安装

推荐使用已经配置好的 conda 环境：

```bash
python
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

本项目主要依赖：

```text
torch
torchvision
numpy
Pillow
matplotlib
```

## 3. 数据集

数据集路径默认是：

```text
./data
```

代码中使用：

```python
torchvision.datasets.Flowers102
```

并采用官方划分：

```text
train / val / test
```

如果数据集路径不同，可以在训练或测试时通过 `--data-root` 指定。

## 4. Baseline 微调实验

任务要求：

- 使用 ImageNet 预训练 ResNet-18；
- 将最后输出层替换为 102 类；
- 新分类层从零开始训练；
- backbone 使用较小学习率微调；
- classifier 使用较大学习率训练。

训练命令：

```bash
python train.py ^
  --experiment task1_baseline_resnet18_pretrained ^
  --model resnet18 ^
  --pretrained ^
  --attention none ^
  --epochs 50 ^
  --batch-size 64 ^
  --num-workers 0 ^
  --lr-backbone 0.0001 ^
  --lr-classifier 0.001
```

测试命令：

```bash
python test.py ^
  --checkpoint checkpoints/task1_baseline_resnet18_pretrained.pth ^
  --model resnet18 ^
  --attention none ^
  --batch-size 64 ^
  --num-workers 0
```

实验结果：

```text
Best Val Acc = 0.7343
Test Acc     = 0.7185
```

## 5. 超参数分析实验

超参数实验主要比较不同学习率、训练轮数和优化器。

### 5.1 SGD 低学习率

```bash
python train.py ^
  --experiment hparam_lr_low ^
  --model resnet18 --pretrained --attention none ^
  --epochs 10 --batch-size 64 --num-workers 0 ^
  --lr-backbone 0.00001 --lr-classifier 0.0001
```

### 5.2 SGD 中学习率

```bash
python train.py ^
  --experiment hparam_lr_mid ^
  --model resnet18 --pretrained --attention none ^
  --epochs 10 --batch-size 64 --num-workers 0 ^
  --lr-backbone 0.0001 --lr-classifier 0.001
```

### 5.3 SGD 高学习率

```bash
python train.py ^
  --experiment hparam_lr_high ^
  --model resnet18 --pretrained --attention none ^
  --epochs 10 --batch-size 64 --num-workers 0 ^
  --lr-backbone 0.0005 --lr-classifier 0.005
```

### 5.4 AdamW 较优组合

```bash
python train.py ^
  --experiment hparam_adamw_20 ^
  --model resnet18 --pretrained --attention none ^
  --epochs 20 --batch-size 64 --num-workers 0 ^
  --optimizer adamw ^
  --lr-backbone 0.0001 --lr-classifier 0.001 ^
  --weight-decay 0.0001
```

对应测试命令示例：

```bash
python test.py ^
  --checkpoint checkpoints/hparam_adamw_20.pth ^
  --model resnet18 ^
  --attention none ^
  --batch-size 64 ^
  --num-workers 0
```

较优结果：

```text
hparam_adamw_20
Best Val Acc = 0.9284
Test Acc     = 0.9034
```

## 6. 预训练消融实验

消融实验中关闭差分学习率，整个网络使用统一学习率。唯一变量是：

```text
是否加载 ImageNet 预训练权重
```

### 6.1 SGD 消融：预训练

```bash
python train.py ^
  --experiment ablation_same_lr_pretrained ^
  --model resnet18 --pretrained --attention none ^
  --epochs 20 --batch-size 64 --num-workers 0 ^
  --optimizer sgd --lr 0.001
```

### 6.2 SGD 消融：随机初始化

```bash
python train.py ^
  --experiment ablation_same_lr_scratch ^
  --model resnet18 --no-pretrained --attention none ^
  --epochs 20 --batch-size 64 --num-workers 0 ^
  --optimizer sgd --lr 0.001
```

### 6.3 AdamW 消融：预训练

```bash
python train.py ^
  --experiment ablation_adamw_same_lr_pretrained ^
  --model resnet18 --pretrained --attention none ^
  --epochs 20 --batch-size 64 --num-workers 0 ^
  --optimizer adamw --lr 0.001 --weight-decay 0.0001
```

### 6.4 AdamW 消融：随机初始化

```bash
python train.py ^
  --experiment ablation_adamw_same_lr_scratch ^
  --model resnet18 --no-pretrained --attention none ^
  --epochs 20 --batch-size 64 --num-workers 0 ^
  --optimizer adamw --lr 0.001 --weight-decay 0.0001
```

消融结果：

| 实验 | Best Val Acc |
| --- | ---: |
| SGD + pretrained | 0.6824 |
| SGD + scratch | 0.1225 |
| AdamW + pretrained | 0.9108 |
| AdamW + scratch | 0.3676 |

曲线图：

```text
report/ablation_val_acc_curve.png
report/ablation_adamw_val_acc_curve.png
```

## 7. 注意力机制实验

注意力模块在 `models.py` 中实现：

- `SEBlock`
- `CBAMBlock`

训练时通过参数控制：

```text
--attention se
--attention cbam
--attention-layers layer4
--attention-layers layer3,layer4
```

### 7.1 SE only layer4

```bash
python train.py ^
  --experiment attn_refine_se_l4_30 ^
  --model resnet18 --pretrained ^
  --attention se --attention-layers layer4 ^
  --epochs 30 --batch-size 64 --num-workers 0 ^
  --optimizer adamw --lr 0.0005 --weight-decay 0.0001
```

### 7.2 CBAM only layer4

```bash
python train.py ^
  --experiment attn_refine_cbam_l4_30 ^
  --model resnet18 --pretrained ^
  --attention cbam --attention-layers layer4 ^
  --epochs 30 --batch-size 64 --num-workers 0 ^
  --optimizer adamw --lr 0.0005 --weight-decay 0.0001
```

### 7.3 SE layer3 + layer4

```bash
python train.py ^
  --experiment attn_refine_se_l34_30 ^
  --model resnet18 --pretrained ^
  --attention se --attention-layers layer3,layer4 ^
  --epochs 30 --batch-size 64 --num-workers 0 ^
  --optimizer adamw --lr 0.0005 --weight-decay 0.0001
```

### 7.4 CBAM layer3 + layer4

```bash
python train.py ^
  --experiment attn_refine_cbam_l34_30 ^
  --model resnet18 --pretrained ^
  --attention cbam --attention-layers layer3,layer4 ^
  --epochs 30 --batch-size 64 --num-workers 0 ^
  --optimizer adamw --lr 0.0005 --weight-decay 0.0001
```

### 7.5 最优结构 + Warmup

```bash
python train.py ^
  --experiment attn_refine_best_se_l34_warmup50 ^
  --model resnet18 --pretrained ^
  --attention se --attention-layers layer3,layer4 ^
  --epochs 50 --warmup-epochs 5 ^
  --batch-size 64 --num-workers 0 ^
  --optimizer adamw --lr 0.0005 --weight-decay 0.0001
```

注意力实验结果：

| 模型 | Best Val Acc | Test Acc |
| --- | ---: | ---: |
| SE layer4 | 0.9235 | 0.9062 |
| CBAM layer4 | 0.9216 | 0.9019 |
| SE layer3+layer4 | 0.9284 | 0.9073 |
| CBAM layer3+layer4 | 0.9196 | 0.8974 |
| SE layer3+layer4 + warmup50 | 0.9294 | 0.9070 |

曲线图：

```text
report/attention_refine_val_acc_curve.png
```

汇总结果：

```text
report/task1_final_summary.csv
```