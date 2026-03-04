import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            # 处理不同的mask文件命名格式
            # 如果mask_ext以'mask_'开头，需要去掉img_id中的'bus_'前缀
            # 例如：img_id='bus_0001-l', mask_ext='mask_.png' -> 'mask_0001-l.png'
            if self.mask_ext.startswith('mask_'):
                # 提取扩展名（去掉'mask_'前缀）
                ext = self.mask_ext[5:]  # 去掉'mask_'（5个字符）
                # 去掉img_id中的'bus_'前缀（如果存在）
                if img_id.startswith('bus_'):
                    img_id_without_prefix = img_id[4:]  # 去掉'bus_'（4个字符）
                else:
                    img_id_without_prefix = img_id
                mask_path = os.path.join(self.mask_dir, str(i), 'mask_' + img_id_without_prefix + ext)
            else:
                # 标准格式：img_id + mask_ext
                mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
            
            # print(mask_path)
            mask.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        if mask.max()<1:
            mask[mask>0] = 1.0
        # print('img:', img.shape, img.dtype, 'mask:', mask.shape, mask.dtype)    
        assert img.size > 0 and mask.size > 0, f"empty sample {idx}: {img.shape} {mask.shape}"

        return img, mask, {'img_id': img_id}
