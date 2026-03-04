"""
PFAN 相关模块（历史实验用）

说明：
- 本项目最终推荐的主模型为 AT-UKanNet（UKAN + ATConv + 并行 CASA）。
- PFAN 相关损失/注意力仅作为历史对比实验保留，默认不启用。
"""
try:
    from .loss import create_criterion, PFAN_AVAILABLE
    __all__ = ['create_criterion', 'PFAN_AVAILABLE']
except ImportError:
    PFAN_AVAILABLE = False
    __all__ = ['PFAN_AVAILABLE']

