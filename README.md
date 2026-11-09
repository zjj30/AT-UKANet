# AT-UKANet

[中文说明](README_ZH.md)

Official code repository for **“AT-UKANet: An Enhanced U-KAN Architecture with Attentive Convolution for Medical Image Segmentation.”**

## Important revision note

The architecture was updated during peer review. In the final manuscript:

- ATConv is used at encoder stages 2 and 3 and the first decoder block;
- dynamic kernels use hybrid global and dilation-2 local context;
- ATConv contains an ungated static 3×3 pathway and a gated dynamic pathway;
- the local-context and dynamic-residual gates are zero-initialized; and
- **Parallel Channel–Spatial Attention (PCSA) is not part of the final model.**

PCSA, channel attention, spatial attention, and the optional edge loss are retained in this repository only as historical or ablation interfaces. Older source comments and compatibility names may still refer to those exploratory configurations; they must not be interpreted as the final paper architecture.

## Final configuration versus legacy interfaces

| Component | Status in the final manuscript |
|---|---|
| ATConv at encoder stages 2–3 | Enabled |
| Channel-matched ATConv at the first decoder block | Enabled |
| Hybrid global/local context | Enabled |
| Static 3×3 pathway | Enabled |
| PCSA / parallel CASA | **Disabled; ablation only** |
| PFAN-style edge loss | Disabled; historical ablation only |

The original `train_atconv_pfan.py` and `scripts_full_experiment_ukan.sh` files are preserved for compatibility with early experiments. In particular, any legacy example that sets `use_attention=True` and `attention_variant=parallel` describes the earlier PCSA ablation, **not** the final AT-UKANet configuration reported in the revised manuscript.

## Repository contents

- `nets/ATConv.py`: attentive convolution operator.
- `nets/archs_ukan_pfan.py`: U-KAN/ATConv experimental architecture and compatibility interfaces.
- `nets/attention_variants.py`: optional attention modules used in ablation experiments.
- `train_atconv_pfan.py`: legacy training entry point.
- `predict_ukan_atconv_pfan.py`: prediction and visualization entry point.
- `data/`, `train/`, and `utils/`: dataset, training, metric, and configuration utilities.

## Environment

Create an isolated Python environment and install the dependencies:

```bash
conda create -n ATUKANet python=3.10
conda activate ATUKANet
pip install -r requirements.txt
```

Install a PyTorch build compatible with the CUDA driver on your system.

## Evaluation protocol reported in the revised manuscript

The revised paper uses nested five-fold evaluation: each outer test fold is held out from model selection, while checkpoint selection uses an inner validation split from the remaining training data. The reported comparison uses matched splits and strong augmentation for AT-UKANet, U-KAN, and Rolling-UNet-L.

The evaluated data versions contain:

| Dataset | Images/pairs |
|---|---:|
| CVC-ClinicDB | 612 |
| BUSI (lesion-only subset) | 647 |
| GlaS | 165 |
| BUSI-WHU | 927 |
| TN3K (harmonized nested split) | 3,493 |
| BUS-BRA | 1,875 |
| BUSI-HF (`data-ous` version) | 996 |

Datasets are not redistributed by this repository. Please obtain them from their original providers and follow their respective licenses and access conditions.

## Reported scope

The revised manuscript does not claim uniform superiority. Under the matched protocol, Rolling-UNet-L has higher mean IoU on all seven datasets. AT-UKANet has lower reported FLOPs and measured batch-1 latency, but more parameters and peak GPU memory. The result is therefore presented as a workload-specific accuracy–resource trade-off.

The five outer folds use one training seed per fold. Reliable patient and center identifiers are unavailable for several datasets, including BUSI-HF; the reported image-level intervals should not be interpreted as patient-level or center-level validation.

## Code-release status

This `main` branch preserves the original public experimental code while correcting the documentation to match the final manuscript. A consolidated paper-aligned reproduction release will be provided as a separate tagged release after its source, protocol files, and third-party licensing inventory have been sanitized for public distribution.

## Citation

Citation metadata will be added after the article record is finalized.

