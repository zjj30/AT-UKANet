"""
将图片数据转换为 .npy 格式，用于 H-vmunet 训练
参考原始 H-vmunet 的数据格式
"""
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import KFold
import argparse


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


def load_images_to_npy(data_dir, dataset_name, img_ids, img_ext, mask_ext, input_h, input_w, num_classes=1):
    """
    加载图片并转换为 numpy 数组
    
    Args:
        data_dir: 数据根目录
        dataset_name: 数据集名称
        img_ids: 图片ID列表
        img_ext: 图片扩展名
        mask_ext: mask扩展名
        input_h: 输入高度
        input_w: 输入宽度
        num_classes: 类别数
    
    Returns:
        tuple: (images, masks) numpy arrays
    """
    images = []
    masks = []
    
    img_dir = os.path.join(data_dir, dataset_name, 'images')
    mask_dir = os.path.join(data_dir, dataset_name, 'masks')
    
    for img_id in tqdm(img_ids, desc=f'Loading {len(img_ids)} images'):
        # 加载图片
        img_path = os.path.join(img_dir, img_id + img_ext)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image: {img_path}")
            continue
        
        # Resize 图片
        img = cv2.resize(img, (input_w, input_h))
        
        # 加载 mask
        mask_list = []
        for i in range(num_classes):
            if mask_ext.startswith('mask_'):
                ext = mask_ext[5:]
                if img_id.startswith('bus_'):
                    img_id_without_prefix = img_id[4:]
                else:
                    img_id_without_prefix = img_id
                mask_path = os.path.join(mask_dir, str(i), 'mask_' + img_id_without_prefix + ext)
            else:
                mask_path = os.path.join(mask_dir, str(i), img_id + mask_ext)
            
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros((input_h, input_w), dtype=np.uint8)
            else:
                mask = np.zeros((input_h, input_w), dtype=np.uint8)
            
            # Resize mask
            mask = cv2.resize(mask, (input_w, input_h))
            mask_list.append(mask)
        
        # 合并多个 mask 通道
        if num_classes == 1:
            mask = mask_list[0]
        else:
            mask = np.stack(mask_list, axis=-1)
        
        images.append(img)
        masks.append(mask)
    
    # 转换为 numpy 数组
    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)
    
    # 归一化图片（与原始 H-vmunet 一致）
    images = dataset_normalized(images)
    
    # 归一化 mask (除以 255)
    masks = masks.astype(np.float32) / 255.0
    
    return images, masks


def prepare_npy_data(data_dir, dataset_name, img_ext, mask_ext, input_h, input_w, 
                     num_classes, k_folds, random_state, output_dir, skip_existing=False):
    """
    准备 .npy 数据文件
    
    Args:
        data_dir: 数据根目录
        dataset_name: 数据集名称
        img_ext: 图片扩展名
        mask_ext: mask扩展名
        input_h: 输入高度
        input_w: 输入宽度
        num_classes: 类别数
        k_folds: k折交叉验证的折数
        random_state: 随机种子
        output_dir: 输出根目录（npy 文件将保存在 {output_dir}/{dataset}/{config}/ 下）
        skip_existing: 如果文件已存在，是否跳过
    """
    # 创建统一的目录结构：{output_dir}/{dataset}/{input_h}x{input_w}_seed{random_state}/
    config_name = f"{input_h}x{input_w}_seed{random_state}"
    dataset_output_dir = os.path.join(output_dir, dataset_name, config_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    print(f"Output directory: {dataset_output_dir}")
    
    # 获取所有图片ID
    img_dir = os.path.join(data_dir, dataset_name, 'images')
    img_paths = sorted(glob(os.path.join(img_dir, '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
    
    print(f"Found {len(img_ids)} images")
    
    # 进行 k-fold 分割
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(img_ids)):
        train_ids = [img_ids[i] for i in train_idx]
        val_ids = [img_ids[i] for i in val_idx]
        
        print(f"\nFold {fold + 1}/{k_folds}:")
        print(f"  Train: {len(train_ids)} samples")
        print(f"  Val: {len(val_ids)} samples")
        
        # 加载训练数据
        print("  Loading train data...")
        train_images, train_masks = load_images_to_npy(
            data_dir, dataset_name, train_ids, img_ext, mask_ext, 
            input_h, input_w, num_classes
        )
        
        # 加载验证数据
        print("  Loading val data...")
        val_images, val_masks = load_images_to_npy(
            data_dir, dataset_name, val_ids, img_ext, mask_ext,
            input_h, input_w, num_classes
        )
        
        # 保存为 .npy 文件
        fold_dir = os.path.join(dataset_output_dir, f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # 检查文件是否已存在
        train_data_path = os.path.join(fold_dir, 'data_train.npy')
        train_mask_path = os.path.join(fold_dir, 'mask_train.npy')
        val_data_path = os.path.join(fold_dir, 'data_val.npy')
        val_mask_path = os.path.join(fold_dir, 'mask_val.npy')
        
        files_exist = all(os.path.exists(p) for p in [train_data_path, train_mask_path, val_data_path, val_mask_path])
        
        if files_exist and skip_existing:
            print(f"  Files already exist, skipping fold {fold + 1}...")
            continue
        
        if files_exist and not skip_existing:
            response = input(f"  Files already exist for fold {fold + 1}. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print(f"  Skipping fold {fold + 1}...")
                continue
        
        # 保存文件
        np.save(train_data_path, train_images)
        np.save(train_mask_path, train_masks)
        np.save(val_data_path, val_images)
        np.save(val_mask_path, val_masks)
        
        print(f"  Saved to {fold_dir}")
        print(f"  Train images shape: {train_images.shape}, masks shape: {train_masks.shape}")
        print(f"  Val images shape: {val_images.shape}, masks shape: {val_masks.shape}")


def main():
    parser = argparse.ArgumentParser(description='Prepare .npy data for H-vmunet')
    parser.add_argument('--data_dir', type=str, required=True, help='Data root directory')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--img_ext', type=str, default='.png', help='Image extension')
    parser.add_argument('--mask_ext', type=str, default='.png', help='Mask extension')
    parser.add_argument('--input_h', type=int, default=256, help='Input height')
    parser.add_argument('--input_w', type=int, default=256, help='Input width')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--random_state', type=int, default=2981, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output root directory for .npy files (default: {data_dir}/npy_data)')
    parser.add_argument('--skip_existing', action='store_true', 
                       help='Skip existing files instead of prompting')
    
    args = parser.parse_args()
    
    # 如果没有指定输出目录，使用默认路径
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'npy_data')
    
    prepare_npy_data(
        args.data_dir, args.dataset, args.img_ext, args.mask_ext,
        args.input_h, args.input_w, args.num_classes,
        args.k_folds, args.random_state, args.output_dir, args.skip_existing
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

