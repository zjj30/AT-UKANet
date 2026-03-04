"""
CBAM风格的轻量级注意力模块
适配UKAN架构
改进点：
1. ChannelAttention: 同时使用 avg 和 max pooling，使用 Conv2d 代替 Linear
2. SpatialAttention: 简化结构，使用通道维度的 avg 和 max pooling
3. 保留原有的类名以保持兼容性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightChannelAttention(nn.Module):
    """
    轻量级通道注意力 (CBAM风格)
    改进点：
    1. 同时使用 avg pooling 和 max pooling
    2. 使用 1x1 卷积代替全连接，更高效
    3. 两个分支共享MLP权重
    """
    def __init__(self, in_channels, reduction=16):
        super(LightweightChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用 1x1 卷积代替全连接，更高效
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        # 两个分支：avg pooling 和 max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # 融合并生成注意力权重，使用广播机制
        return x * self.sigmoid(avg_out + max_out)


class LightweightSpatialAttention(nn.Module):
    """
    轻量级空间注意力 (CBAM风格)
    改进点：
    1. 使用通道维度的 avg 和 max pooling
    2. 简化网络结构，参数量很小
    3. 只需一个 7x7 卷积
    """
    def __init__(self, kernel_size=7):
        super(LightweightSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        # 在通道维度上做平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        # 生成空间注意力图，使用广播机制
        return x * self.sigmoid(self.conv(concat))


class CBAMBlock(nn.Module):
    """
    CBAM注意力块
    可以单独使用CA或SA，也可以串联使用（CBAM标准做法是先CA后SA）
    """
    def __init__(self, in_channels, use_ca=True, use_sa=True, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.use_ca = use_ca
        self.use_sa = use_sa
        
        if use_ca:
            self.ca = LightweightChannelAttention(in_channels, reduction)
        if use_sa:
            self.sa = LightweightSpatialAttention(kernel_size)

    def forward(self, x):
        """
        CBAM 的标准做法是串联：先CA后SA
        但支持单独使用
        """
        if self.use_ca:
            x = self.ca(x)
        if self.use_sa:
            x = self.sa(x)
        return x


# ==================== 兼容层：保留原有的类名 ====================
# 为了保持与现有代码的兼容性，保留原来的实现和类名

class ChannelWiseAttention(nn.Module):
    """
    原始的 Channel-wise Attention (CA) 模块
    保留用于兼容性
    """
    def __init__(self, in_channels, reduction=4):
        super(ChannelWiseAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Global Average Pooling + FC layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.size()

        # Global Average Pooling
        y = self.avg_pool(x).view(B, C)  # (B, C)

        # FC layers
        y = self.fc1(y)  # (B, C//reduction)
        y = self.relu(y)
        y = self.fc2(y)  # (B, C)
        y = self.sigmoid(y)  # (B, C)

        # Reshape and multiply
        y = y.view(B, C, 1, 1)  # (B, C, 1, 1)
        y = y.expand_as(x)  # (B, C, H, W)

        out = x * y
        return out


class SpatialAttention(nn.Module):
    """
    原始的 Spatial Attention (SA) 模块
    保留用于兼容性
    """
    def __init__(self, in_channels, k=7):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.k = k

        # 第一个分支: (1, k) -> (k, 1)
        self.conv1_1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, k), padding=(0, k//2))
        self.bn1_1 = nn.BatchNorm2d(in_channels // 2)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels // 2, 1, kernel_size=(k, 1), padding=(k//2, 0))
        self.bn1_2 = nn.BatchNorm2d(1)
        self.relu1_2 = nn.ReLU(inplace=True)

        # 第二个分支: (k, 1) -> (1, k)
        self.conv2_1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(k, 1), padding=(k//2, 0))
        self.bn2_1 = nn.BatchNorm2d(in_channels // 2)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels // 2, 1, kernel_size=(1, k), padding=(0, k//2))
        self.bn2_2 = nn.BatchNorm2d(1)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            attention: (B, C, H, W) - 扩展到与输入相同的通道数
        """
        # 第一个分支: (1, k) -> (k, 1)
        att1 = self.conv1_1(x)
        att1 = self.bn1_1(att1)
        att1 = self.relu1_1(att1)
        att1 = self.conv1_2(att1)
        att1 = self.bn1_2(att1)
        att1 = self.relu1_2(att1)

        # 第二个分支: (k, 1) -> (1, k)
        att2 = self.conv2_1(x)
        att2 = self.bn2_1(att2)
        att2 = self.relu2_1(att2)
        att2 = self.conv2_2(att2)
        att2 = self.bn2_2(att2)
        att2 = self.relu2_2(att2)

        # 相加并sigmoid
        attention = att1 + att2  # (B, 1, H, W)
        attention = self.sigmoid(attention)

        # 扩展到与输入相同的通道数（使用expand而不是expand_as，更高效）
        attention = attention.expand(x.size(0), x.size(1), x.size(2), x.size(3))  # (B, C, H, W)

        return attention


class PFANAttentionBlock(nn.Module):
    """
    原始的 PFAN注意力模块组合
    保留用于兼容性
    """
    def __init__(self, in_channels, use_ca=True, use_sa=True, reduction=4, k=9):
        super(PFANAttentionBlock, self).__init__()
        self.use_ca = use_ca
        self.use_sa = use_sa

        if use_ca:
            self.ca = ChannelWiseAttention(in_channels, reduction=reduction)
        if use_sa:
            self.sa = SpatialAttention(in_channels, k=k)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        if self.use_ca:
            x = self.ca(x)

        if self.use_sa:
            sa_map = self.sa(x)
            x = x * sa_map

        return x
