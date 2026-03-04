"""
学习率调度器模块
"""
from torch.optim import lr_scheduler


def create_scheduler(optimizer, config):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置字典
    
    Returns:
        scheduler: 学习率调度器（可能为None）
    """
    if config['scheduler'] == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config['factor'], 
            patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[int(e) for e in config['milestones'].split(',')], 
            gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        return None
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented")

