"""
H-vmunet 数据加载器（使用 .npy 文件）
参考原始 H-vmunet 的 loader.py
"""
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from scipy import ndimage


def dataset_normalized(imgs):
    """
    数据集归一化（与原始 H-vmunet 一致）
    """
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_min = np.min(imgs_normalized[i])
        imgs_max = np.max(imgs_normalized[i])
        if imgs_max - imgs_min > 0:
            imgs_normalized[i] = ((imgs_normalized[i] - imgs_min) / (imgs_max - imgs_min)) * 255
        else:
            imgs_normalized[i] = 0
    return imgs_normalized


class HvmunetLoader(Dataset):
    """
    H-vmunet 数据加载器（使用 .npy 文件）
    参考原始 H-vmunet 的 isic_loader
    """
    def __init__(self, data_path, train=True, test=False):
        """
        Args:
            data_path: .npy 文件所在目录（包含 data_train.npy, mask_train.npy 等）
            train: 是否为训练集
            test: 是否为测试集（如果为 True，使用 val 数据作为 test）
        """
        super(HvmunetLoader, self).__init__()
        self.train = train
        self.test = test
        
        if train:
            self.data = np.load(data_path + 'data_train.npy')
            self.mask = np.load(data_path + 'mask_train.npy')
        else:
            if test:
                self.data = np.load(data_path + 'data_test.npy')
                self.mask = np.load(data_path + 'mask_test.npy')
            else:
                self.data = np.load(data_path + 'data_val.npy')
                self.mask = np.load(data_path + 'mask_val.npy')
        
        # 数据归一化（与原始 H-vmunet 一致）
        self.data = dataset_normalized(self.data)
        
        # Mask 处理：扩展维度（如果 mask 已经在 prepare_npy_data.py 中归一化，这里不需要再除以 255）
        # 但为了与原始代码一致，我们检查 mask 的值域
        if len(self.mask.shape) == 3:
            self.mask = np.expand_dims(self.mask, axis=3)
        
        # 如果 mask 值域在 [0, 255]，则归一化到 [0, 1]
        if self.mask.max() > 1.0:
            self.mask = self.mask / 255.0
    
    def __getitem__(self, idx):
        img = self.data[idx]
        seg = self.mask[idx]
        
        if self.train:
            # 训练时的数据增强
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)
        
        # 转换为 tensor
        seg = torch.tensor(seg.copy(), dtype=torch.float32)
        img = torch.tensor(img.copy(), dtype=torch.float32)
        
        # 调整维度顺序：HWC -> CHW
        if len(img.shape) == 3:
            img = img.permute(2, 0, 1)
        if len(seg.shape) == 3:
            seg = seg.permute(2, 0, 1)
        
        return img, seg
    
    def random_rot_flip(self, image, label):
        """随机旋转和翻转"""
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label
    
    def random_rotate(self, image, label):
        """随机旋转"""
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label
    
    def __len__(self):
        return len(self.data)

