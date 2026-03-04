## AT-UKanNet: UKAN + ATConv + 并行 CASA

### 🔍 项目目标

本项目最终提出的主模型为 **AT-UKanNet**，用于医学图像分割等场景，通过在 UKAN 主干上引入
**ATConv** 与 **并行 CASA 注意力（Channel & Spatial Attention）**，增强全局上下文建模能力。

- **主干网络**: UKAN
- **卷积增强 (ATConv)**: 在编码器/解码器部分层替换标准卷积，提升非线性表达和上下文建模能力
- **注意力模块 (CASA)**: 基于 CBAM 思路实现的通道注意力 + 空间注意力，其中推荐方案为 **并行 CASA**
- **可选损失增强**: 保留 PFAN 风格边界保持损失（Edge Preservation Loss）作为历史对比实验，默认关闭

> 说明：ARConv 与 PFAN 注意力在本项目中的效果一般，已作为**历史方案**保留，仅用于 ablation / 对比，不再作为最终推荐模块。

---

## ⚙️ 环境准备

推荐使用 Conda 新建独立环境，并通过 `requirements.txt` 安装依赖：

```bash
conda create -n ATUKANet python=3.10
conda activate ATUKANet
pip install -r requirements.txt
```

---

## 🧱 模型结构概览

- **AT-UKanNet 实现**: `nets/archs_ukan_pfan.py` 中的 `AT_UKanNet` 类  
  - 兼容别名：`UKAN_PFAN = AT_UKanNet`（为了不破坏旧脚本）
- **ATConv 卷积模块**: `nets/ATConv.py` 中的 `ATConv2d`
- **并行 CASA 注意力模块**: `nets/attention_variants.py`
  - `ParallelCASABlock`（核心模块）
  - 工厂函数 `create_attention_block(variant='parallel', ...)`

主要控制开关（均在配置 / 命令行参数中暴露）：

- **+1 `use_atconv`**: 是否在 encoder/decoder 指定层启用 ATConv  
- **+2 `use_attention` + `attention_variant`**: 是否启用注意力模块（推荐 `parallel` = 并行 CASA）  
- **+3 `use_hybrid_arch`**: 是否在解码端与注意力做更深层次结构融合（默认 False，仅保留接口）  
- **+4 `use_edge_loss`**: 是否启用 PFAN 风格 Edge Preservation Loss（默认 False，仅历史对比）  

---

## 🧪 训练与消融实验

### 1. 基本训练脚本（AT-UKanNet）

主训练入口为根目录下的 `train_atconv_pfan.py`，负责：

- 解析配置：`utils/config.py`
- K 折训练流程：`train/kfold_trainer.py`
- 模型创建：`nets/model_factory.py`（内部使用 `AT_UKanNet`）

示例（单次实验）：

```bash
python train_atconv_pfan.py \
  --datasets bus_bar \
  --data_dir /path/to/data \
  --output_dir outputs \
  --arch AT-UKanNet \
  --use_atconv True \
  --atconv_encoder_layers 3 \
  --atconv_decoder_layers none \
  --use_attention True \
  --attention_variant parallel \
  --use_hybrid_arch False \
  --use_edge_loss False
```

### 2. 批量完整实验脚本

`scripts_full_experiment_ukan.sh` 会对多个数据集、随机种子与折数进行批量实验调度：

- 模型名：`MODEL_NAME="AT_UKanNet"`
- Attention 变体：`ATTENTION_VARIANT`（推荐 `parallel`）
- K 折：通过 `K_FOLDS` 与 `FOLD_TO_RUN` 控制

在 Linux 环境下运行：

```bash
bash scripts_full_experiment_ukan.sh
```

---

## 🔎 预测与可视化

- **批量预测**: `predict_ukan_atconv_pfan.py`
  - 会加载已训练好的 AT-UKanNet 权重（历史目录名中仍可能包含 `UKAN_ATConv_PFAN`）
  - 支持指定数据集与样本 ID，并保存预测 mask、可视化结果与可选 Grad-CAM

示例：

```bash
python predict_ukan_atconv_pfan.py \
  --dataset bus_bra \
  --model_dir /path/to/exp_dir \
  --img_ids bus_0001-l,bus_0259-l \
  --output_dir predicted_masks \
  --generate_gradcam
```

---

## 📁 主要目录结构（简要）

```text
.
├── nets/                 # 模型相关模块
│   ├── archs_ukan_pfan.py    # AT_UKanNet (主模型，实现 UKAN + ATConv + CASA)
│   ├── ATConv.py             # ATConv2d 与示例模块
│   ├── attention_variants.py # CASA/CBAM 注意力变体（并行 CASA 为主）
│   ├── model_factory.py      # 统一模型工厂（创建 AT-UKanNet）
│   └── optimizer.py, scheduler.py, kan.py, ...
├── train/                # 训练流程（k-fold、训练、验证、指标）
├── data/                 # 数据集加载与 k-fold 划分
├── utils/                # 配置解析、随机种子、设备等工具
├── pfan/                 # PFAN 风格损失（历史对比实验用）
├── train_atconv_pfan.py  # AT-UKanNet 训练主脚本
├── scripts_full_experiment_ukan.sh  # AT-UKanNet 批量实验脚本
└── predict_ukan_atconv_pfan.py      # AT-UKanNet 预测与可视化脚本
```

---

## 📝 备注与历史说明

- **AT-UKanNet** 是本项目最终推荐的模型，基于 UKAN 主干 + ATConv + 并行 CASA。
- **PFAN 注意力 / ARConv**：已在项目中做过尝试，但效果不及 AT-UKanNet 的设计，
  因此仅作为历史对比选项保留在代码与配置中。



