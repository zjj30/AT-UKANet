"""
AT-UKanNet: UKAN + ATConv + 并行 CASA 注意力 整体架构

最终采用的主模型为 **AT-UKanNet**：
- 主干：UKAN
- 卷积增强：ATConv 替换部分编码器/解码器卷积层（+1）
- 注意力：并行 CASA（Channel & Spatial Attention，并行通道/空间注意力模块，基于并行 CBAM 实现）（+2）
- 结构：可选的解码端混合架构（+3）
- 损失：可选的 PFAN 风格边界保持损失 Edge Preservation Loss（+4，在 train 模块中实现，默认关闭，仅保留为对比实验）

说明：
- 原 PFAN 注意力模块作为历史尝试保留在 `pfan_attention.py` 中，但不再作为默认方案。
- 推荐配置：use_atconv=True, use_attention=True, attention_variant='parallel'（并行 CASA）。
"""
import torch
from torch import nn
import torch.nn.functional as F
import math
import sys
import os

# 添加父目录到路径，以便导入模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 添加当前目录到路径，以便导入同目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from kan import KANLinear, KAN
from ATConv import ATConv2d
from pfan_attention import ChannelWiseAttention, SpatialAttention, PFANAttentionBlock, CBAMBlock
from attention_variants import create_attention_block

__all__ = ['AT_UKanNet', 'UKAN_PFAN', 'D_ConvLayer', 'ConvLayer', 'ATConvLayer',
           'PatchEmbed', 'DW_bn_relu', 'DWConv', 'KANBlock', 'KANLayer']


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc3 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, H, W)
    
        return x


class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ✅ 修复后的ATConv卷积层
class ATConvLayer(nn.Module):
    """
    ATConv卷积层，用于替换UKAN中的标准卷积层

    修复说明：
    1. ATConv2d的forward只接受一个参数(x)
    2. 将BN和activation设置为外部控制
    3. 保留epoch和hw_range参数用于将来可能的扩展
    """
    def __init__(self, in_ch, out_ch, epoch=0, hw_range=[3, 7]):
        super(ATConvLayer, self).__init__()
        self.epoch = epoch
        self.hw_range = hw_range

        # ✅ 使用ATConv替换第一个卷积，禁用内部BN和activation
        self.atconv1 = ATConv2d(
            in_ch, out_ch,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            use_bn=False,      # 在外部使用BN
            activation=None    # 在外部使用ReLU
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        # ✅ 使用ATConv替换第二个卷积
        self.atconv2 = ATConv2d(
            out_ch, out_ch,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            use_bn=False,
            activation=None
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input):
        # ✅ 修复：ATConv2d的forward只需要输入x
        x = self.atconv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.atconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

    def update_epoch(self, epoch):
        """更新epoch参数（保留接口用于将来扩展）"""
        self.epoch = epoch


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class AT_UKanNet(nn.Module):
    """
    AT-UKanNet 主模型：UKAN + ATConv + 并行 CASA 注意力

    支持的消融实验开关：
    - use_atconv (+1): 在编码器/解码器指定层使用 ATConv，替换标准卷积
    - use_attention (+2): 在特征提取阶段启用并行 CASA 注意力模块（基于并行 CBAM 的 Channel + Spatial Attention）
    - use_hybrid_arch (+3): 在解码端引入与注意力结合的混合架构（当前默认关闭，仅保留接口）
    - use_edge_loss (+4): 在损失函数层面引入 PFAN 风格的边界保持损失（在 train 模块中实现，默认关闭）

    说明：
    - 早期 PFAN 注意力（ChannelWiseAttention + SpatialAttention）在本项目中效果一般，已作为历史方案保留。
    - 当前推荐配置为：use_atconv=True, use_attention=True 且 attention_variant='parallel'（并行 CASA）。
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16,
                 in_chans=3, embed_dims=[256, 320, 512], no_kan=False, drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[1, 1, 1], use_atconv=True,
                 hw_range_min=3, hw_range_max=7, atconv_encoder_layers='3', atconv_decoder_layers='none',
                 use_attention=False, use_hybrid_arch=False, attention_variant='serial', **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]
        self.use_atconv = use_atconv  # +1: 是否使用 ATConv
        self.use_attention = use_attention  # +2: 是否在特征提取阶段启用并行 CASA 注意力
        self.use_hybrid_arch = use_hybrid_arch  # +3: 是否启用解码端混合架构
        self.attention_variant = attention_variant  # 注意力变体类型（推荐: 'parallel' 并行 CASA）
        self.current_epoch = 0
        self.hw_range = [hw_range_min, hw_range_max]

        # ✅ 解析ATConv层配置
        self.atconv_encoder_layers = self._parse_layer_config(
            atconv_encoder_layers, max_layers=3
        )
        self.atconv_decoder_layers = self._parse_layer_config(
            atconv_decoder_layers, max_layers=5
        )

        # 打印配置信息，便于记录不同实验设置
        print(f"\n{'='*60}")
        print(f"AT-UKanNet Configuration:")
        print(f"  use_atconv: {use_atconv}")
        print(f"  hw_range: {self.hw_range}")
        print(f"  ATConv encoder layers: {self.atconv_encoder_layers if self.atconv_encoder_layers else 'None'}")
        print(f"  ATConv decoder layers: {self.atconv_decoder_layers if self.atconv_decoder_layers else 'None'}")
        print(f"  use_attention: {use_attention}")
        if use_attention:
            print(f"  attention_variant: {attention_variant}")
        print(f"  use_hybrid_arch: {use_hybrid_arch}")
        print(f"{'='*60}\n")

        # ✅ 编码器层 - 根据配置决定是否使用ATConv
        print("Building Encoder:")
        self.encoder1 = self._create_conv_layer(
            1, 3, kan_input_dim // 8,
            use_atconv and 1 in self.atconv_encoder_layers
        )
        self.encoder2 = self._create_conv_layer(
            2, kan_input_dim//8, kan_input_dim//4,
            use_atconv and 2 in self.atconv_encoder_layers
        )
        self.encoder3 = self._create_conv_layer(
            3, kan_input_dim // 4, kan_input_dim,
            use_atconv and 3 in self.atconv_encoder_layers
        )
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=no_kan
        )])
        
        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, no_kan=no_kan
        )])
        
        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=no_kan
        )])
        
        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, no_kan=no_kan
        )])
        
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        
        # +2: 根据变体创建相应的注意力模块
        if use_attention:
            if attention_variant == 'multiscale':
                # 多尺度变体：Stage4 和 Bottleneck 都使用注意力
                self.attention_multiscale = create_attention_block(
                    'multiscale',
                    embed_dims[2],  # bottleneck_channels
                    stage4_channels=embed_dims[1],      # Stage4: 320 channels
                    reduction=16
                )
                print(f"✓ Using Multi-Scale CBAM: Stage4({embed_dims[1]}) + Bottleneck({embed_dims[2]})")
                print(f"  - Variant: {attention_variant}")
            else:
                # 其他变体：只在 Bottleneck 使用注意力
                if attention_variant == 'pyramid':
                    # Pyramid变体不需要reduction参数
                    self.attention_bottleneck = create_attention_block(
                        attention_variant,
                        embed_dims[2]  # 512 channels
                    )
                else:
                    self.attention_bottleneck = create_attention_block(
                        attention_variant,
                embed_dims[2],  # 512 channels
                        reduction=16
            )
                print(f"✓ Using {attention_variant.upper()} attention at Bottleneck ({embed_dims[2]} channels)")
                print(f"  - Variant: {attention_variant}, Reduction: 16")
        
        # ✅ 解码器层 - 支持灵活配置
        print("\nBuilding Decoder:")
        self.decoder1 = self._create_decoder_layer(
            1, embed_dims[2], embed_dims[1],
            use_atconv and 1 in self.atconv_decoder_layers
        )
        self.decoder2 = self._create_decoder_layer(
            2, embed_dims[1], embed_dims[0],
            use_atconv and 2 in self.atconv_decoder_layers
        )
        self.decoder3 = self._create_decoder_layer(
            3, embed_dims[0], embed_dims[0]//4,
            use_atconv and 3 in self.atconv_decoder_layers
        )
        self.decoder4 = self._create_decoder_layer(
            4, embed_dims[0]//4, embed_dims[0]//8,
            use_atconv and 4 in self.atconv_decoder_layers
        )
        self.decoder5 = self._create_decoder_layer(
            5, embed_dims[0]//8, embed_dims[0]//8,
            use_atconv and 5 in self.atconv_decoder_layers
        )
        
        # +3: 方案A中不使用decoder注意力（可在后续实验中启用）
        if use_hybrid_arch:
            print("⚠ Note: use_hybrid_arch is enabled but not used in Plan A")
            print("  Plan A focuses on Bottleneck CA only for minimal improvement")
        
        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

        print(f"\n{'='*60}\n")

    def _parse_layer_config(self, config_str, max_layers):
        """
        解析层配置字符串

        参数:
            config_str: 配置字符串 ('none', 'all', '1,2,3')
            max_layers: 最大层数

        返回:
            层号列表 [1, 2, 3] 或 []
        """
        if not config_str or config_str.lower() == 'none':
            return []
        elif config_str.lower() == 'all':
            return list(range(1, max_layers + 1))
        else:
            try:
                layers = [int(x.strip()) for x in config_str.split(',') if x.strip()]
                # 验证层号有效性
                valid_layers = [l for l in layers if 1 <= l <= max_layers]
                if len(valid_layers) != len(layers):
                    print(f"Warning: Some layer numbers in '{config_str}' are out of range (1-{max_layers})")
                return valid_layers
            except ValueError:
                print(f"Warning: Invalid layer config '{config_str}', using 'none'")
                return []

    def _create_conv_layer(self, layer_num, in_ch, out_ch, use_atconv):
        """
        创建编码器层 (ConvLayer或ATConvLayer)

        参数:
            layer_num: 层号
            in_ch: 输入通道数
            out_ch: 输出通道数
            use_atconv: 是否使用ATConv
        """
        if use_atconv:
            print(f"  Encoder{layer_num}: ATConvLayer ({in_ch} -> {out_ch})")
            return ATConvLayer(in_ch, out_ch, epoch=0, hw_range=self.hw_range)
        else:
            print(f"  Encoder{layer_num}: ConvLayer ({in_ch} -> {out_ch})")
            return ConvLayer(in_ch, out_ch)

    def _create_decoder_layer(self, layer_num, in_ch, out_ch, use_atconv):
        """
        创建解码器层 (D_ConvLayer或ATConvLayer)

        参数:
            layer_num: 层号
            in_ch: 输入通道数
            out_ch: 输出通道数
            use_atconv: 是否使用ATConv
        """
        if use_atconv:
            print(f"  Decoder{layer_num}: ATConvLayer ({in_ch} -> {out_ch})")
            return ATConvLayer(in_ch, out_ch, epoch=0, hw_range=self.hw_range)
        else:
            print(f"  Decoder{layer_num}: D_ConvLayer ({in_ch} -> {out_ch})")
            return D_ConvLayer(in_ch, out_ch)

    def update_epoch(self, epoch):
        """更新所有ATConv层的epoch参数"""
        self.current_epoch = epoch

        # 更新编码器中的ATConv层
        for encoder in [self.encoder1, self.encoder2, self.encoder3]:
            if hasattr(encoder, 'update_epoch'):
                encoder.update_epoch(epoch)

        # 更新解码器中的ATConv层
        for decoder in [self.decoder1, self.decoder2, self.decoder3,
                       self.decoder4, self.decoder5]:
            if hasattr(decoder, 'update_epoch'):
                decoder.update_epoch(epoch)

    def forward(self, x):
        B = x.shape[0]
        
        ### Encoder - Conv Stage
        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        ### Tokenized KAN Stage
        ### Stage 4
        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Stage 4 (before Bottleneck)
        # +2: Multiscale变体在Stage4也应用注意力
        if self.use_attention and self.attention_variant == 'multiscale':
            t4 = self.attention_multiscale.forward_stage4(t4)

        ### Bottleneck
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # +2: 在 Bottleneck 应用注意力
        if self.use_attention:
            if self.attention_variant == 'multiscale':
                out = self.attention_multiscale.forward_bottleneck(out)
            else:
                out = self.attention_bottleneck(out)

        ### Decoder Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2,2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2,2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2,2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2,2), mode='bilinear'))

        return self.final(out)

    def get_config_summary(self):
        """返回模型配置摘要"""
        atconv_count = 0
        total_conv_layers = 8  # 3 encoder + 5 decoder

        # 统计ATConv层数量
        for layer in [self.encoder1, self.encoder2, self.encoder3,
                     self.decoder1, self.decoder2, self.decoder3,
                     self.decoder4, self.decoder5]:
            if isinstance(layer, ATConvLayer):
                atconv_count += 1

        return {
            'total_conv_layers': total_conv_layers,
            'atconv_layers': atconv_count,
            'standard_conv_layers': total_conv_layers - atconv_count,
            'atconv_encoder_layers': self.atconv_encoder_layers,
            'atconv_decoder_layers': self.atconv_decoder_layers,
            'hw_range': self.hw_range,
            'use_attention': self.use_attention,
            'use_hybrid_arch': self.use_hybrid_arch
        }


# 兼容旧代码的别名：UKAN_PFAN -> AT_UKanNet
# 这样不会破坏已有脚本/配置，但推荐在新代码中直接使用 AT_UKanNet 名称。
UKAN_PFAN = AT_UKanNet

