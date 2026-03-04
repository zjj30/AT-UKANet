"""
PFAN 损失函数模块（历史对比实验）

说明：
- 早期曾尝试在主干模型上叠加 PFAN 提出的 EdgeHoldLoss 等边界保持损失；
  实验结果显示提升有限，因此在 AT-UKanNet 中 **默认不启用**。
- 该模块仅在配置 `use_edge_loss=True` 且成功导入 PFAN 依赖时才会生效，
  适合作为论文/报告中的对比实验项。
"""
import torch.nn as nn
import sys
import os

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import losses

# 尝试导入外部 PFAN 损失函数实现（若环境中不存在，则退化为不可用状态）
try:
    from UKAN_PFAN.pfan_loss import EdgeHoldLoss, CombinedLoss
    PFAN_AVAILABLE = True
except ImportError:
    print("Warning: PFAN loss not available, edge loss will be disabled")
    PFAN_AVAILABLE = False
    CombinedLoss = None


def create_criterion(config, device):
    """
    根据配置创建损失函数
    
    Args:
        config: 配置字典
        device: 设备
    
    Returns:
        criterion: 损失函数
    """
    # 如果使用edge loss且PFAN可用
    if config.get('use_edge_loss') and PFAN_AVAILABLE:
        # 创建基础损失函数
        if config['loss'] == 'BCEWithLogitsLoss':
            base_loss = nn.BCEWithLogitsLoss().to(device)
        else:
            base_loss = losses.__dict__[config['loss']]().to(device)
        
        # 创建组合损失函数
        edge_loss_weight = config.get('edge_loss_weight', 0.15)
        criterion = CombinedLoss(
            base_loss,
            use_edge_loss=True,
            edge_loss_weight=edge_loss_weight,
            saliency_pos=1.12,
            edge_weight=0.3,
            saliency_weight=0.7
        ).to(device)
        print(f"Using Combined Loss: {config['loss']} + EdgeHoldLoss (weight: {edge_loss_weight})")
    else:
        # 使用标准损失函数
        if config['loss'] == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss().to(device)
        elif config['loss'] == 'HybridBoundaryLoss':
            criterion = losses.HybridBoundaryLoss().to(device)
        else:
            criterion = losses.__dict__[config['loss']]().to(device)
    
    return criterion

