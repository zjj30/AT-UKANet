import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple

# ATConv: Attentive convolution: Unifying the expressivity of self-attention with convolutional efficiency

class ATConv2d(nn.Module):
    """
    Attentive Convolution (UNet drop-in-friendly)
    - Generates per-sample per-output convolution kernels from global context (C2K)
    - Applies differential kernel modulation (DKM): kernel = kernel - lambda * mean(kernel)
    - Performs convolution via unfold + batched matmul (works for arbitrary in/out channels)
    Notes:
      * This implementation produces per-sample dynamic weights -> uses unfold (not F.conv2d).
      * For large images / channels consider memory/time implications; production use may need optimized CUDA kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        context_channels: Optional[int] = None,
        reduction: int = 4,
        use_bn: bool = False,
        activation: Optional[Callable] = nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = kernel_size
        self.stride = stride
        self.padding = (kernel_size // 2) if padding is None else padding
        self.bias_flag = bias
        self.use_bn = use_bn
        self.activation = activation

        # context projection: 1x1 conv to extract context features
        ctx_ch = context_channels or max(8, out_channels // reduction)
        self.context_proj = nn.Conv2d(in_channels, ctx_ch, kernel_size=1, bias=True)

        # kernel generator: map context vector -> flattened kernels
        # Produces per-sample weights of shape (out_channels * in_channels * K * K)
        self.kernel_gen = nn.Linear(ctx_ch, out_channels * in_channels * self.K * self.K)

        # differential modulation parameter per output channel (learnable)
        self.gamma = nn.Parameter(torch.zeros(out_channels))  # will pass through sigmoid

        # optional bias per output channel
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # value / output projection (1x1) optional to mimic value projection + mixing
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # optional BN + activation - useful to mimic common UNet blocks
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        # initialize kernel_gen small
        nn.init.normal_(self.kernel_gen.weight, std=1e-3)
        nn.init.zeros_(self.kernel_gen.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        returns: (B, C_out, H_out, W_out)  (H_out/W_out depend on stride/padding)
        """
        B, C_in, H, W = x.shape
        assert C_in == self.in_channels, f"in_channels mismatch: got {C_in} expect {self.in_channels}"

        # 1) Context-to-Kernel translation (C2K)
        ctx = self.context_proj(x)                  # (B, ctx_ch, H, W)
        # global context pooling -> per-sample vector
        ctx_vec = F.adaptive_avg_pool2d(ctx, (1, 1)).view(B, -1)  # (B, ctx_ch)

        # generate kernels: (B, out_channels * in_channels * K * K)
        kernels_flat = self.kernel_gen(ctx_vec)     # (B, O*C*KK)
        kernels = kernels_flat.view(B, self.out_channels, self.in_channels, self.K, self.K)
        # apply small nonlinearity if desired (paper uses φ, e.g., GELU/Tanh). Use tanh here to stabilize.
        kernels = torch.tanh(kernels)

        # 2) Differential Kernel Modulation (DKM) - lateral inhibition
        # compute mean per kernel (per sample, per out_channel)
        mean_k = kernels.view(B, self.out_channels, -1).mean(dim=-1, keepdim=True)  # (B, O, 1)
        # gamma per out_channel -> (1, O, 1, 1, 1)
        lam = torch.sigmoid(self.gamma).view(1, self.out_channels, 1, 1, 1)
        # subtract scaled mean
        kernels = kernels - lam * mean_k.view(B, self.out_channels, 1, 1, 1)

        # 3) Value projection (optional) + unfold -> batched matmul
        v = self.value_proj(x) if hasattr(self, "value_proj") else x  # (B, C_in, H, W)

        # Use unfold to extract sliding local patches
        # patches: (B, C_in * K * K, L)   where L = H_out * W_out
        patches = F.unfold(v, kernel_size=self.K, dilation=1, padding=self.padding, stride=self.stride)
        # reshape kernels for batched matmul:
        # kernels: (B, O, C_in, K, K) -> (B, O, C_in * K * K)
        kernels_mat = kernels.view(B, self.out_channels, -1)  # (B, O, Ck)
        # perform batched matmul: out[b] = kernels_mat[b] @ patches[b]  => (O, L)
        # we can do via einsum or bmm after reshape
        # reshape patches to (B, Ck, L)
        # result (B, O, L)
        out = torch.einsum('boc,bcl->bol', kernels_mat, patches)  # (B, O, L)

        # add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)

        # reshape to (B, out_channels, H_out, W_out)
        H_out = (H + 2 * self.padding - self.K) // self.stride + 1
        W_out = (W + 2 * self.padding - self.K) // self.stride + 1
        out = out.view(B, self.out_channels, H_out, W_out)

        # optional BN + activation
        if self.bn is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.activation(out)

        return out


# -----------------------
# Example: drop-in replace Conv2d in a UNet double-conv block
# -----------------------
class DoubleConvUNetBlock(nn.Module):
    """
    Typical UNet double conv block:
      (Conv -> BN -> ReLU) x 2
    Replace Conv with ATConv2d to test plugin behaviour.
    """
    def __init__(self, in_ch, out_ch, use_atconv=True):
        super().__init__()
        Conv = ATConv2d if use_atconv else nn.Conv2d
        # first conv
        self.conv1 = Conv(in_ch, out_ch, kernel_size=3, padding=1, bias=True, use_bn=True, activation=nn.ReLU(inplace=True))
        # second conv
        self.conv2 = Conv(out_ch, out_ch, kernel_size=3, padding=1, bias=True, use_bn=True, activation=nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# =========== quick smoke test ===========
if __name__ == "__main__":
    input = torch.randn(2, 3, 128, 128)   # e.g. image batch
    at = ATConv2d(in_channels=3, out_channels=16, kernel_size=3, use_bn=True)
    out = at(input)
    print("in:", input.shape, "out:", out.shape)
    # Try as UNet block
    block = DoubleConvUNetBlock(3, 16, use_atconv=True)
    o2 = block(input)
    print("double conv out:", o2.shape)
