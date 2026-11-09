# AT-UKANet

[English README](README.md)

本仓库对应论文 **“AT-UKANet: An Enhanced U-KAN Architecture with Attentive Convolution for Medical Image Segmentation”**。

## 重要的二修说明

论文最终模型在审稿过程中进行了调整：

- ATConv 位于编码器第 2、3 阶段以及第一个解码模块；
- 动态卷积核同时使用全局上下文和 dilation=2 的局部上下文；
- ATConv 包含不带门控的静态 3×3 分支和带门控的动态分支；
- 局部上下文门与动态残差门采用零初始化；
- **Parallel Channel–Spatial Attention（PCSA）不属于最终模型。**

PCSA、通道注意力、空间注意力和可选边界损失仅作为历史实验或消融接口保留。旧源码中的部分注释、兼容名称和实验开关可能仍提及这些探索性配置，但不能将其理解为论文最终架构。

## 最终配置与旧接口的区别

| 组件 | 论文最终状态 |
|---|---|
| 编码器第 2–3 阶段 ATConv | 启用 |
| 第一个解码模块的通道匹配 ATConv | 启用 |
| 全局/局部混合上下文 | 启用 |
| 静态 3×3 分支 | 启用 |
| PCSA / parallel CASA | **关闭，仅用于消融** |
| PFAN 风格边界损失 | 关闭，仅为历史消融 |

`train_atconv_pfan.py` 与 `scripts_full_experiment_ukan.sh` 是早期实验的兼容入口。旧示例中设置 `use_attention=True`、`attention_variant=parallel` 的命令对应早期 PCSA 消融，**不是二修论文报告的最终 AT-UKANet 配置**。

## 仓库内容

- `nets/ATConv.py`：注意力卷积算子。
- `nets/archs_ukan_pfan.py`：U-KAN/ATConv 实验结构及兼容接口。
- `nets/attention_variants.py`：消融实验使用的可选注意力模块。
- `train_atconv_pfan.py`：旧训练入口。
- `predict_ukan_atconv_pfan.py`：预测和可视化入口。
- `data/`、`train/`、`utils/`：数据、训练、指标和配置工具。

## 环境

```bash
conda create -n ATUKANet python=3.10
conda activate ATUKANet
pip install -r requirements.txt
```

请根据本机 CUDA 驱动安装兼容的 PyTorch 版本。

## 二修论文的评估协议

论文采用嵌套五折评估：每个外层测试折不参与模型选择，checkpoint 只根据其余训练数据中的内层验证集确定。AT-UKANet、U-KAN 和 Rolling-UNet-L 使用一致的数据划分和强数据增强。

论文使用的数据版本为：CVC 612、BUSI 病灶子集 647、GlaS 165、BUSI-WHU 927、TN3K 3493、BUS-BRA 1875，以及 BUSI-HF 的 `data-ous` 996 对版本。数据集不随本仓库再分发，请从原始提供方获取并遵守相应许可。

## 结果边界

二修论文不再声称模型普遍优于所有对照。在统一协议下，Rolling-UNet-L 在 7 个数据集上的平均 IoU 均高于 AT-UKANet；AT-UKANet 的 FLOPs 和实测 batch-1 延迟更低，但参数量与峰值显存更高。因此论文将其解释为特定工作负载下的准确率—资源权衡。

每个外层折仅使用一个训练随机种子。包括 BUSI-HF 在内的多个数据集缺少可靠的患者和中心标识，因此图像级区间不能解释为患者级或中心级验证。

## 代码发布状态

当前 `main` 分支保留早期公开实验代码，并已将文档修正为与最终论文一致。完整的论文复现版本将在源码、协议文件及第三方许可清单完成公开发布审计后，以独立 tag 发布。

## 引用

论文正式出版后将补充引用信息。

