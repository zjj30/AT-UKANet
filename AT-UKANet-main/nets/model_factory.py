"""
模型创建工厂

当前默认支持的主架构为 AT-UKanNet（UKAN + ATConv + 并行 CASA 注意力），
可以正确处理 use_atconv / use_attention / use_hybrid_arch 等消融实验参数。
"""
import sys
import os

# 添加当前目录到路径，以便导入同目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import archs_ukan_pfan

# 主模型类（推荐使用的新名称）
AT_UKanNet = archs_ukan_pfan.AT_UKanNet
# 兼容旧名称（UKAN_PFAN）——内部依然可用，但不再在文档中作为主名出现
UKAN_PFAN = archs_ukan_pfan.UKAN_PFAN


def create_model(config):
    """
    根据配置创建模型

    Args:
        config: 配置字典

    Returns:
        model: 创建的 AT-UKanNet 模型
    """
    # 使用 AT-UKanNet 架构（UKAN + ATConv + 并行 CASA 注意力）
    model = AT_UKanNet(
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        img_size=config['input_h'],
        patch_size=16,
        in_chans=config['input_channels'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan'],
        # ATConv 相关参数（+1）
        use_atconv=config.get('use_atconv', True),              # 是否使用 ATConv，默认使用
        hw_range_min=config.get('hw_range_min', 3),             # ATConv 的 hw_range 最小值
        hw_range_max=config.get('hw_range_max', 7),             # ATConv 的 hw_range 最大值
        atconv_encoder_layers=config.get('atconv_encoder_layers', '3'),   # 编码器中使用 ATConv 的层
        atconv_decoder_layers=config.get('atconv_decoder_layers', 'none'),# 解码器中使用 ATConv 的层
        # 并行 CASA 注意力相关参数（+2）
        use_attention=config.get('use_attention', False),       # 是否启用注意力机制
        use_hybrid_arch=config.get('use_hybrid_arch', False),   # 是否启用解码端混合架构（+3）
        attention_variant=config.get('attention_variant', 'parallel')      # 注意力变体类型（推荐 'parallel' 并行 CASA）
    )

    return model

