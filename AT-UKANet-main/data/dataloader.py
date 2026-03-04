"""
数据加载器模块
"""
import os
import torch
from albumentations.augmentations import transforms, geometric
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize

# 导入Dataset（从当前data目录）
import sys
import importlib.util

# 使用绝对路径导入dataset模块
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'dataset.py')
spec = importlib.util.spec_from_file_location("dataset", dataset_path)
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
Dataset = dataset_module.Dataset


def create_fold_dataloaders(config):
    """
    为单折创建数据加载器
    
    Args:
        config: 配置字典，需要包含：
            - train_ids, val_ids: 训练和验证的图片ID列表
            - data_dir, dataset: 数据目录和数据集名称
            - img_ext, mask_ext: 图片和mask扩展名
            - input_h, input_w: 输入尺寸
            - batch_size: 批次大小
            - num_workers: 数据加载器工作进程数
            - num_classes: 类别数
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # 训练集数据增强
    train_transform = Compose([
        RandomRotate90(),
        geometric.transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # 验证集数据增强（仅resize和normalize）
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # 创建数据集
    train_dataset = Dataset(
        img_ids=config['train_ids'],
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    
    val_dataset = Dataset(
        img_ids=config['val_ids'],
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    
    return train_loader, val_loader

