"""
CASA 注意力机制（Channel & Spatial Attention）的改进变体
专门针对 UKAN / AT-UKanNet 的全局特征捕捉弱点设计。

实现上基于 CBAM 风格的通道注意力 + 空间注意力，但在命名与论文描述中统一称为
CASA（Channel and Spatial Attention），其中推荐的主方案为**并行 CASA**。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 基础组件 ====================

class LightweightChannelAttention(nn.Module):
    """轻量级通道注意力 (CASA / CBAM 风格)"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)) #全局平均池化 得到通道重要性权重
        max_out = self.fc(self.max_pool(x)) #全局最大池化 得到通道重要性权重
        return x * self.sigmoid(avg_out + max_out) #用学习到的通道重要性权重，对原始特征图的每个通道进行差异化调控


class LightweightSpatialAttention(nn.Module):
    """轻量级空间注意力 (CASA / CBAM 风格)"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


# ==================== 方案 0: 基线（串联 CASA） ====================

class SerialCASABlock(nn.Module):
    """
    方案0: 串联 CASA (基线)
    CA → SA 串联
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = LightweightChannelAttention(in_channels, reduction)
        self.sa = LightweightSpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)  # 先通道注意力
        x = self.sa(x)  # 再空间注意力
        return x



# ==================== 方案 1: 并行 CASA ====================

class ParallelCASABlock(nn.Module):
    """
    方案2: 并行 CASA
    CA 和 SA 并行计算，避免信息损失
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = LightweightChannelAttention(in_channels, reduction)
        self.sa = LightweightSpatialAttention(kernel_size)

    def forward(self, x):
        # 并行计算
        ca_out = self.ca(x)  # CA 作用于原始输入
        sa_out = self.sa(x)  # SA 也作用于原始输入
        # 相乘融合
        return ca_out * sa_out

# ==================== 工厂函数 ====================

def create_attention_block(variant, in_channels, **kwargs):
    """
    创建指定变体的注意力模块（目前主推并行 CASA）
    
    Args:
        variant: 变体名称 ('parallel')
        in_channels: 输入通道数
        **kwargs: 额外参数
    """
    if variant == 'parallel':
        # 并行 CASA：Channel Attention 与 Spatial Attention 并行计算后相乘融合
        return ParallelCASABlock(in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ========== 向后兼容别名（旧代码中使用的 CBAM 命名） ==========
SerialCBAMBlock = SerialCASABlock
ParallelCBAMBlock = ParallelCASABlock


# ==================== 测试函数 ====================

def test_attention_variants():
    """测试所有注意力变体"""
    B, C, H, W = 2, 256, 14, 14
    x = torch.randn(B, C, H, W)
    
    variants = {
        'Serial CASA': SerialCASABlock(C),
        'Parallel CASA': ParallelCASABlock(C),
    }
    
    print("="*60)
    print("Testing Attention Variants")
    print("="*60)
    
    for name, module in variants.items():
        module.eval()
        with torch.no_grad():
            out = module(x)
        
        params = sum(p.numel() for p in module.parameters())
        print(f"\n{name}:")
        print(f"  Input:  {x.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Params: {params:,}")
        assert out.shape == x.shape, f"{name} shape mismatch!"
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    test_attention_variants()

