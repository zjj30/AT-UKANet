# H-vmunet 数据准备说明

## 概述

为了与原始 H-vmunet 代码保持一致，我们使用 .npy 文件格式来加载数据。这需要先将图片数据转换为 .npy 格式。

## 步骤 1: 准备 .npy 数据

运行以下命令将图片数据转换为 .npy 格式：

```bash
# 进入 AT-UKanNet 项目根目录（本仓库所在目录）
cd /mnt/a/weather-ZJJ/ZhuJJ/U-KAN-main/Seg_UKAN/AT-UKanNet   # 示例路径，可按实际位置修改

python data/prepare_npy_data.py \
    --data_dir /mnt/a/weather-ZJJ/ZhuJJ/UKan/inputs \
    --dataset busi \
    --img_ext .png \
    --mask_ext .png \
    --input_h 256 \
    --input_w 256 \
    --num_classes 1 \
    --k_folds 5 \
    --random_state 2981 \
    --output_dir /mnt/a/weather-ZJJ/ZhuJJ/UKan/inputs/npy_data
```

**注意**：如果不指定 `--output_dir`，默认会在 `{data_dir}/npy_data` 下创建。

### 参数说明

- `--data_dir`: 数据根目录（包含数据集文件夹）
- `--dataset`: 数据集名称（如 busi）
- `--img_ext`: 图片扩展名（如 .png）
- `--mask_ext`: mask扩展名（如 .png）
- `--input_h`: 输入图片高度（默认 256）
- `--input_w`: 输入图片宽度（默认 256）
- `--num_classes`: 类别数（默认 1）
- `--k_folds`: k折交叉验证的折数（默认 5）
- `--random_state`: 随机种子（默认 2981）
- `--output_dir`: 输出根目录（默认：{data_dir}/npy_data）
- `--skip_existing`: 如果文件已存在，自动跳过（不提示）

### 输出结构

转换完成后，会在输出目录下创建以下结构（按数据集、配置和折数组织）：

```
{output_dir}/
└── {dataset}/
    └── {input_h}x{input_w}_seed{random_state}/
        ├── fold0/
        │   ├── data_train.npy
        │   ├── mask_train.npy
        │   ├── data_val.npy
        │   └── mask_val.npy
        ├── fold1/
        │   ├── data_train.npy
        │   ├── mask_train.npy
        │   ├── data_val.npy
        │   └── mask_val.npy
        ...
```

**示例**：
```
/mnt/a/weather-ZJJ/ZhuJJ/UKan/inputs/npy_data/
└── busi/
    └── 256x256_seed2981/
        ├── fold0/
        ├── fold1/
        ├── fold2/
        ├── fold3/
        └── fold4/
```

### 复用说明

- **相同配置可复用**：如果使用相同的数据集、输入尺寸和随机种子，npy 文件可以复用
- **自动检测**：如果文件已存在，脚本会提示是否覆盖，或使用 `--skip_existing` 自动跳过
- **目录组织**：按配置自动组织目录，便于管理和查找

## 步骤 2: 修改训练脚本配置

在训练脚本中，可以添加以下配置（可选）：

```python
config = {
    # ... 其他配置 ...
    'use_npy_data': True,  # 使用 .npy 数据（默认 True）
    'npy_data_dir': None,  # .npy 数据根目录（默认：自动从 data_dir 推断）
    'use_hvmunet_loss': True,  # 使用原始 H-vmunet 的损失函数（默认 True）
    'threshold': 0.5,  # 验证时的二值化阈值（默认 0.5）
}
```

**自动路径推断**：
- 如果不指定 `npy_data_dir`，训练脚本会自动根据以下参数构建路径：
  - `data_dir`: 数据根目录
  - `dataset`: 数据集名称
  - `input_h`, `input_w`: 输入尺寸
  - `dataseed`: 随机种子
- 自动路径格式：`{data_dir}/npy_data/{dataset}/{input_h}x{input_w}_seed{dataseed}/`

**手动指定路径**：
- 如果 npy 文件保存在其他位置，可以手动指定 `npy_data_dir`，指向 `{dataset}/{config}/` 目录

## 主要改动

1. **数据加载方式**：
   - 使用 `HvmunetLoader` 从 .npy 文件加载数据
   - 数据已经预处理和归一化（与原始 H-vmunet 一致）

2. **损失函数**：
   - 使用原始 H-vmunet 的 `BceDiceLoss`
   - 需要模型输出先经过 sigmoid（使用 BCELoss，不是 BCEWithLogitsLoss）

3. **验证指标计算**：
   - 使用展平后计算 IoU 的方式（与原始代码一致）
   - 使用混淆矩阵计算 IoU 和 Dice

## 注意事项

1. 确保在运行训练前已经生成了 .npy 文件
2. `npy_data_dir` 应该指向包含 `fold0/`, `fold1/` 等文件夹的目录
3. 数据归一化方式与原始 H-vmunet 完全一致

