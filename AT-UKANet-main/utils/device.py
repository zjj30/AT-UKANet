"""
设备管理工具
"""
import torch


def get_device(config):
    """根据config获取设备"""
    if config['gpu_ids'] != '-1':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device('cpu')

