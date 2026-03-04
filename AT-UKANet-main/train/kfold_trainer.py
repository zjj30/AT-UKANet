"""
k折训练管理模块
"""
import os
import yaml
import torch
import pandas as pd
from collections import OrderedDict
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

# 添加父目录和当前目录到路径
import sys
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(current_dir)

# 确保当前目录优先
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 使用明确的导入方式，避免与父目录的utils.py冲突
import importlib.util

def import_module_from_path(module_name, file_path):
    """从文件路径导入模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入当前目录下的模块
utils_device = import_module_from_path("utils_device", os.path.join(current_dir, "utils", "device.py"))
get_device = utils_device.get_device

data_dataloader = import_module_from_path("data_dataloader", os.path.join(current_dir, "data", "dataloader.py"))
create_fold_dataloaders = data_dataloader.create_fold_dataloaders

nets_model_factory = import_module_from_path("nets_model_factory", os.path.join(current_dir, "nets", "model_factory.py"))
create_model = nets_model_factory.create_model

nets_optimizer = import_module_from_path("nets_optimizer", os.path.join(current_dir, "nets", "optimizer.py"))
setup_optimizer = nets_optimizer.setup_optimizer

nets_scheduler = import_module_from_path("nets_scheduler", os.path.join(current_dir, "nets", "scheduler.py"))
create_scheduler = nets_scheduler.create_scheduler

pfan_loss = import_module_from_path("pfan_loss", os.path.join(current_dir, "pfan", "loss.py"))
create_criterion = pfan_loss.create_criterion

train_trainer = import_module_from_path("train_trainer", os.path.join(current_dir, "train", "trainer.py"))
train = train_trainer.train

train_validator = import_module_from_path("train_validator", os.path.join(current_dir, "train", "validator.py"))
validate = train_validator.validate


def run_single_fold_experiment(config, exp_suffix):
    """
    运行单折实验
    
    Args:
        config: 配置字典
        exp_suffix: 实验后缀（如 "fold0"）
    
    Returns:
        dict: 实验结果
    """
    output_dir = config['output_dir']
    device = get_device(config)
    
    exp_name = f"{config['name']}_{exp_suffix}"
    # 创建主实验目录（不含fold后缀）
    main_exp_dir = os.path.join(output_dir, config['name'])
    os.makedirs(main_exp_dir, exist_ok=True)

    # 创建fold子目录
    exp_dir = os.path.join(main_exp_dir, exp_suffix)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    config_to_save = config.copy()
    config_to_save.pop('train_ids', None)
    config_to_save.pop('val_ids', None)
    with open(f'{exp_dir}/config.yml', 'w') as f:
        yaml.dump(config_to_save, f)
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    
    # 创建优化器
    optimizer = setup_optimizer(model, config)
    
    # 创建损失函数
    criterion = create_criterion(config, device)
    
    # 创建调度器
    scheduler = create_scheduler(optimizer, config)
    
    # 创建数据加载器
    train_loader, val_loader = create_fold_dataloaders(config)
    
    # 训练循环
    best_metrics = train_fold(
        config, train_loader, val_loader, model, criterion, optimizer, scheduler, device, exp_dir
    )
    
    # 清理
    del model, optimizer, criterion
    if scheduler is not None:
        del scheduler
    del train_loader, val_loader
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return {
        'exp_name': exp_name,
        'fold': config['current_fold'],
        'metrics': best_metrics
    }


def train_fold(config, train_loader, val_loader, model, criterion, optimizer, scheduler, device, exp_dir):
    """
    训练单折的函数
    
    Args:
        config: 配置字典
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 模型
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        exp_dir: 实验目录
    
    Returns:
        dict: 最佳指标
    """
    my_writer = SummaryWriter(exp_dir)
    
    log = OrderedDict([
        ('epoch', []), ('lr', []), ('loss', []), ('iou', []),
        ('val_loss', []), ('val_iou', []), ('val_dice', []),
        ('val_recall', []), ('val_precision', []), ('val_specificity', []), ('val_f1', [])
    ])
    
    best_iou = 0
    best_dice = 0
    best_f1 = 0
    best_metrics = {}
    final_metrics = {}  # 保存最后一个epoch的指标
    trigger = 0
    
    for epoch in range(config['epochs']):
        print(f'Fold {config["current_fold"]+1} - Epoch [{epoch+1}/{config["epochs"]}]')

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer, epoch, device)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion, device)

        if config['scheduler'] == 'CosineAnnealingLR' and scheduler is not None:
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau' and scheduler is not None:
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_f1 %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['f1']))

        # 记录日志
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_recall'].append(val_log['recall'])
        log['val_precision'].append(val_log['precision'])
        log['val_specificity'].append(val_log['specificity'])
        log['val_f1'].append(val_log['f1'])

        pd.DataFrame(log).to_csv(f'{exp_dir}/log.csv', index=False)

        # TensorBoard记录
        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)
        my_writer.add_scalar('val/recall', val_log['recall'], global_step=epoch)
        my_writer.add_scalar('val/precision', val_log['precision'], global_step=epoch)
        my_writer.add_scalar('val/specificity', val_log['specificity'], global_step=epoch)
        my_writer.add_scalar('val/f1', val_log['f1'], global_step=epoch)
        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)

        trigger += 1

        # 保存最佳模型
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{exp_dir}/model_best_iou.pth')
            best_iou = val_log['iou']
            best_metrics = val_log.copy()
            best_metrics['best_iou'] = best_iou
            print("=> saved best IoU model")
            print(f'IoU: %.4f' % best_iou)
            trigger = 0

        if val_log['dice'] > best_dice:
            best_dice = val_log['dice']
            best_metrics['best_dice'] = best_dice

        if val_log['f1'] > best_f1:
            torch.save(model.state_dict(), f'{exp_dir}/model_best_f1.pth')
            best_f1 = val_log['f1']

        # 保存最后一个epoch的指标
        final_metrics = val_log.copy()

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
    
    my_writer.close()
    
    # 确保返回的指标包含所有需要的字段
    if best_metrics:
        result_metrics = best_metrics.copy()
    else:
        # 如果best_metrics为空（不应该发生，但为了安全），使用final_metrics
        result_metrics = final_metrics.copy() if final_metrics else {}
        result_metrics['best_iou'] = best_iou
    
    result_metrics['best_iou'] = best_iou
    result_metrics['best_dice'] = best_dice
    result_metrics['final_iou'] = final_metrics.get('iou', 0) if final_metrics else 0
    result_metrics['final_dice'] = final_metrics.get('dice', 0) if final_metrics else 0
    
    return result_metrics

