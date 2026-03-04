"""
训练循环模块
"""
from collections import OrderedDict
from tqdm import tqdm

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


def train(config, train_loader, model, criterion, optimizer, epoch, device):
    """
    训练一个epoch
    
    Args:
        config: 配置字典
        train_loader: 训练数据加载器
        model: 模型
        criterion: 损失函数
        optimizer: 优化器
        epoch: 当前epoch
        device: 设备
    
    Returns:
        OrderedDict: 训练指标 {'loss': ..., 'iou': ...}
    """
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}

    model.train()
    
    # 更新ATConv的epoch参数
    if hasattr(model, 'module') and hasattr(model.module, 'update_epoch'):
        model.module.update_epoch(epoch)
    elif hasattr(model, 'update_epoch'):
        model.update_epoch(epoch)

    # pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()

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
        
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        # postfix = OrderedDict([
        #     ('loss', avg_meters['loss'].avg),
        #     ('iou', avg_meters['iou'].avg),
        # ])
    #     pbar.set_postfix(postfix)
    #     pbar.update(1)
    # pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg)
    ])

