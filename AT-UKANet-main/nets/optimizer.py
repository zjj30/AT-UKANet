"""
优化器设置模块
"""
import torch.optim as optim


def setup_optimizer(model, config):
    """
    设置优化器
    
    Args:
        model: 模型
        config: 配置字典
    
    Returns:
        optimizer: 优化器
    """
    param_groups = []
    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower():
            kan_fc_params.append(param)
        else:
            other_params.append(param)

    if kan_fc_params:
        param_groups.append({
            'params': kan_fc_params,
            'lr': config['kan_lr'],
            'weight_decay': config['kan_weight_decay']
        })
    
    param_groups.append({
        'params': other_params,
        'lr': config['lr'],
        'weight_decay': config['weight_decay']
    })

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, momentum=config['momentum'], nesterov=config['nesterov'])
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} not implemented")
    
    return optimizer

