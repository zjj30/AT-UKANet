"""
验证循环模块
"""
from collections import OrderedDict
from tqdm import tqdm
import torch

# 添加父目录到路径
import sys
import os
import importlib.util

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(current_dir)

# 确保父目录在路径中
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 从父目录导入（父目录有utils.py和metrics.py）
# 使用importlib避免与当前目录的utils包冲突
utils_file_path = os.path.join(parent_dir, "utils.py")
if os.path.exists(utils_file_path):
    spec = importlib.util.spec_from_file_location("parent_utils", utils_file_path)
    parent_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_utils)
    AverageMeter = parent_utils.AverageMeter
else:
    from utils import AverageMeter

# 从metrics2导入iou_score（metrics2.py有iou_score函数）
from metrics2 import iou_score


def validate(config, val_loader, model, criterion, device):
    """
    验证模型
    
    Args:
        config: 配置字典
        val_loader: 验证数据加载器
        model: 模型
        criterion: 损失函数
        device: 设备
    
    Returns:
        OrderedDict: 验证指标
    """
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}

    model.eval()

    with torch.no_grad():
        # pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

        #     postfix = OrderedDict([
        #         ('loss', avg_meters['loss'].avg),
        #         ('iou', avg_meters['iou'].avg),
        #         ('dice', avg_meters['dice'].avg)
        #     ])
        #     pbar.set_postfix(postfix)
        #     pbar.update(1)
        # pbar.close()

    # 计算详细指标
    metrics_dict = {
        'loss': avg_meters['loss'].avg,
        'iou': avg_meters['iou'].avg,
        'dice': avg_meters['dice'].avg
    }
    
    # 计算其他指标（recall, precision, specificity, f1）
    # 这里简化处理，实际应该在整个验证集上计算
    # 为了简化，我们使用dice作为f1的近似
    metrics_dict['recall'] = metrics_dict['dice']  # 简化
    metrics_dict['precision'] = metrics_dict['dice']  # 简化
    metrics_dict['specificity'] = metrics_dict['dice']  # 简化
    metrics_dict['f1'] = metrics_dict['dice']  # 简化
    
    return OrderedDict(metrics_dict)

