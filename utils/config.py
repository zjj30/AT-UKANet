"""
配置解析模块
"""
import argparse
import sys
import os

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 定义 str2bool 函数（避免循环导入）
def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ['true', '1', 'yes', 'y', 't']:
        return True
    elif v.lower() in ['false', '0', 'no', 'n', 'f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 定义 list_type 函数（避免循环导入）
def list_type(s):
    """将逗号分隔的字符串转换为列表"""
    if isinstance(s, list):
        return s
    return [item.strip() for item in s.split(',') if item.strip()]

# 延迟导入，避免循环依赖
def get_loss_names():
    """获取损失函数名称列表"""
    import losses
    loss_names = list(losses.__all__)
    loss_names.append('BCEWithLogitsLoss')
    return loss_names

def get_arch_names():
    """获取架构名称列表"""
    # 延迟导入主模型类，避免循环依赖
    from nets.archs_ukan_pfan import AT_UKanNet
    # 返回可选架构名称列表（目前主推 AT-UKanNet）
    return ['AT_UKanNet']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='data split seed')
    
    # GPU设置
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='comma separated list of GPU ids to use')
    
    # k折交叉验证参数
    parser.add_argument('--k_folds', default=5, type=int,
                        help='number of folds for k-fold cross validation (default: 5)')
    parser.add_argument('--fold_to_run', default="all", type=str,
                        help='specific fold(s) to run, e.g., "0,1,2" or "all" (default: all)')
    
    # 数据集参数
    parser.add_argument('--datasets', default='busi', type=str,
                        help='dataset name')
    
    # ATConv相关参数 - +1
    parser.add_argument('--use_atconv', default=True, type=str2bool,
                        help='whether to use ATConv')
    parser.add_argument('--atconv_encoder_layers', default='3', type=str,
                        help='which encoder layers to use ATConv (e.g., "3" or "all" or "none")')
    parser.add_argument('--atconv_decoder_layers', default='none', type=str,
                        help='which decoder layers to use ATConv (e.g., "1" or "all" or "none")')
    parser.add_argument('--hw_range_min', default=3, type=int,
                        help='ATConv hw_range minimum value')
    parser.add_argument('--hw_range_max', default=7, type=int,
                        help='ATConv hw_range maximum value')
    
    # 注意力 / 损失改进参数 - 消融实验配置
    parser.add_argument('--use_attention', default=False, type=str2bool,
                        help='+2: whether to use CASA attention (parallel channel+spatial attention, 基于并行 CBAM) in feature extraction')
    parser.add_argument('--attention_variant', default='serial', type=str,
                        choices=['serial', 'multiscale', 'parallel', 'global', 'hybrid', 'pyramid'],
                        help='Attention variant type: serial (baseline), multiscale, parallel, global, hybrid, pyramid')
    parser.add_argument('--use_hybrid_arch', default=False, type=str2bool,
                        help='+3: whether to use hybrid architecture with attention in decoder')
    parser.add_argument('--use_edge_loss', default=False, type=str2bool,
                        help='+4: whether to use PFAN-style edge preservation loss (历史尝试，默认 False)')
    parser.add_argument('--edge_loss_weight', default=0.15, type=float,
                        help='weight for EdgeHoldLoss when combined with BCEDiceLoss (default: 0.15)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='AT-UKanNet',
                        help='model architecture name (default: AT-UKanNet)')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss and optimizer
    loss_names = get_loss_names()
    parser.add_argument('--loss', default='BCEDiceLoss', choices=loss_names,
                        help='loss: ' + ' | '.join(loss_names) + ' (default: BCEDiceLoss)')
    parser.add_argument('--data_dir', default=r'/home/deploy/ZhuJJ/UKan/inputs', 
                        help='dataset dir')
    parser.add_argument('--output_dir', default='outputs', help='output dir')
    
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='optimizer: Adam | SGD (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')
    
    parser.add_argument('--kan_lr', default=1e-2, type=float, metavar='LR', 
                        help='initial learning rate for KAN layers')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay for KAN layers')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', 
                        help='early stopping (default: -1), -1: means no early stopping')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--no_kan', action='store_true')

    config = parser.parse_args()
    return config

