"""
模型和数据集配置
用于批量预测和保存 mask / 可视化不同模型（含 AT-UKanNet 与若干对比模型）。

说明：
- `ours` / `ours_xxx` 对应本项目最终模型 **AT-UKanNet**（UKAN + ATConv + 并行 CASA）。
- `unet_arconv_xxx` 为早期基于 ARConv 的改进 UNet，对性能贡献有限，仅作为历史对比实验保留。
"""

# 模型配置格式: (model_name, exp_dir, dataset, use_arconv)
# 其中 use_arconv 标记该配置是否属于 ARConv 系列（历史对比模型，非推荐方案）。
MODEL_CONFIGS = [
    # GLAS数据集
    ("unet_baseline", "", "glas", False),
    ("att_unet", "outputs/att_unet_glas", "glas", False),
    ("unetpp", "outputs/unetpp_glas", "glas", False),
    ("rolling_unet", "outputs/rolling_unet_glas", "glas", False),
    ("u_mamba", "outputs/u_mamba_glas", "glas", False),
    ("u_kan", "/mnt/a/weather-ZJJ/ZhuJJ/U-KAN-main/Seg_UKAN/UKAN_ATConv_PFAN/outputs/20260108_094515_glas_UKAN_seed8142_fold0", "glas", False),
    ("ours", "/mnt/a/weather-ZJJ/ZhuJJ/U-KAN-main/Seg_UKAN/UKAN_ATConv_PFAN/experiments_cbam_variants/outputs/20260108_224244_UKAN_ATConv_glas_seed6142_fold0", "glas", False),

    # BUSI数据集
    ("unet_baseline_busi", "outputs/unet_baseline_busi", "busi", False),
    ("unet_arconv_busi", "outputs/unet_arconv_busi", "busi", True),
    ("att_unet_busi", "outputs/att_unet_busi", "busi", False),
    ("unetpp_busi", "outputs/unetpp_busi", "busi", False),
    ("rolling_unet_busi", "outputs/rolling_unet_busi", "busi", False),
    ("u_mamba_busi", "outputs/u_mamba_busi", "busi", False),
    ("u_kan_busi", "outputs/u_kan_busi", "busi", False),
    ("ours_busi", "outputs/ours_busi", "busi", False),

    # BUS_BRA数据集
    ("unet_baseline_bus_bra", "outputs/unet_baseline_bus_bra", "bus_bra", False),
    ("unet_arconv_bus_bra", "outputs/unet_arconv_bus_bra", "bus_bra", True),
    ("att_unet_bus_bra", "outputs/att_unet_bus_bra", "bus_bra", False),
    ("unetpp_bus_bra", "outputs/unetpp_bus_bra", "bus_bra", False),
    ("rolling_unet_bus_bra", "outputs/rolling_unet_bus_bra", "bus_bra", False),
    ("u_mamba_bus_bra", "outputs/u_mamba_bus_bra", "bus_bra", False),
    ("u_kan_bus_bra", "outputs/u_kan_bus_bra", "bus_bra", False),
    ("ours_bus_bra", "outputs/ours_bus_bra", "bus_bra", False),

    # OURS数据集
    ("unet_baseline_ours", "outputs/unet_baseline_ours", "ours", False),
    ("unet_arconv_ours", "outputs/unet_arconv_ours", "ours", True),
    ("att_unet_ours", "outputs/att_unet_ours", "ours", False),
    ("unetpp_ours", "outputs/unetpp_ours", "ours", False),
    ("rolling_unet_ours", "outputs/rolling_unet_ours", "ours", False),
    ("u_mamba_ours", "outputs/u_mamba_ours", "ours", False),
    ("u_kan_ours", "outputs/u_kan_ours", "ours", False),
    ("ours_ours", "outputs/ours_ours", "ours", False),

    # BUSI_WHU数据集
    ("unet_baseline_busi_whu", "outputs/unet_baseline_busi_whu", "busi_whu", False),
    ("unet_arconv_busi_whu", "outputs/unet_arconv_busi_whu", "busi_whu", True),
    ("att_unet_busi_whu", "outputs/att_unet_busi_whu", "busi_whu", False),
    ("unetpp_busi_whu", "outputs/unetpp_busi_whu", "busi_whu", False),
    ("rolling_unet_busi_whu", "outputs/rolling_unet_busi_whu", "busi_whu", False),
    ("u_mamba_busi_whu", "outputs/u_mamba_busi_whu", "busi_whu", False),
    ("u_kan_busi_whu", "outputs/u_kan_busi_whu", "busi_whu", False),
    ("ours_busi_whu", "outputs/ours_busi_whu", "busi_whu", False),

    # TN3K数据集
    ("unet_baseline_TN3K", "outputs/unet_baseline_TN3K", "TN3K", False),
    ("unet_arconv_TN3K", "outputs/unet_arconv_TN3K", "TN3K", True),
    ("att_unet_TN3K", "outputs/att_unet_TN3K", "TN3K", False),
    ("unetpp_TN3K", "outputs/unetpp_TN3K", "TN3K", False),
    ("rolling_unet_TN3K", "outputs/rolling_unet_TN3K", "TN3K", False),
    ("u_mamba_TN3K", "outputs/u_mamba_TN3K", "TN3K", False),
    ("u_kan_TN3K", "outputs/u_kan_TN3K", "TN3K", False),
    ("ours_TN3K", "outputs/ours_TN3K", "TN3K", False),

    # CVC数据集
    ("unet_baseline_cvc", "outputs/unet_baseline_cvc", "cvc", False),
    ("unet_arconv_cvc", "outputs/unet_arconv_cvc", "cvc", True),
    ("att_unet_cvc", "outputs/att_unet_cvc", "cvc", False),
    ("unetpp_cvc", "outputs/unetpp_cvc", "cvc", False),
    ("rolling_unet_cvc", "outputs/rolling_unet_cvc", "cvc", False),
    ("u_mamba_cvc", "outputs/u_mamba_cvc", "cvc", False),
    ("u_kan_cvc", "outputs/u_kan_cvc", "cvc", False),
    ("ours_cvc", "outputs/ours_cvc", "cvc", False),
]

# 测试图片ID配置（可以根据数据集选择有代表性的图片）
TEST_IMG_IDS = {
    "glas": ["5", "15", "134"],  # 每个数据集选3个图片
    "busi": ["benign (1)", "benign (45)", "malignant (15)"],  # 选择有代表性的样本
    "bus_bra": ["case001", "case045", "case089"],
    "ours": ["img001", "img045", "img089"],
    "busi_whu": ["img001", "img045", "img089"],  # 武汉BUSI数据集
    "TN3K": ["img001", "img045", "img089"],  # TN3K数据集
    "cvc": ["img001", "img045", "img089"],  # CVC数据集
}

# 默认测试图片（如果没有指定数据集特定的图片）
DEFAULT_IMG_IDS = [
    "img001", "img045", "img089", "img123", "img156", "img200",
    "img250", "img300", "img350", "img400", "img450", "img500"
]

def get_model_configs_by_dataset(dataset_name: str = None):
    """
    获取指定数据集的所有模型配置

    Args:
        dataset_name: 数据集名称，如果为None则返回所有配置

    Returns:
        模型配置列表
    """
    if dataset_name is None:
        return MODEL_CONFIGS

    return [config for config in MODEL_CONFIGS if config[2] == dataset_name]

def get_img_ids_by_dataset(dataset_name: str):
    """
    获取指定数据集的测试图片ID

    Args:
        dataset_name: 数据集名称

    Returns:
        图片ID列表
    """
    return TEST_IMG_IDS.get(dataset_name, DEFAULT_IMG_IDS)

def get_available_datasets():
    """获取所有可用的数据集名称"""
    return list(TEST_IMG_IDS.keys())

def get_available_models():
    """获取所有可用的模型名称"""
    return list(set(config[0] for config in MODEL_CONFIGS))

# 便捷函数：按模型类型分组
def get_configs_by_model_type():
    """按模型类型分组返回配置"""
    model_types = {
        "unet_baseline": [],
        "unet_arconv": [],
        "att_unet": [],
        "unetpp": [],
        "rolling_unet": [],
        "u_mamba": [],
        "u_kan": [],
        "ours": []
    }

    for config in MODEL_CONFIGS:
        model_name = config[0].split('_')[0]  # 取第一个下划线前的部分
        if model_name in model_types:
            model_types[model_name].append(config)

    return model_types