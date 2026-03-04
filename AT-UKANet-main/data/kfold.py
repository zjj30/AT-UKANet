"""
k折交叉验证模块
"""
import os
import sys
import importlib.util
from glob import glob
from sklearn.model_selection import KFold


def get_kfold_splits(img_ids, k_folds=5, random_state=42):
    """
    生成k折交叉验证的数据分割
    
    Args:
        img_ids: 图片ID列表
        k_folds: 折数
        random_state: 随机种子
    
    Returns:
        list: [(fold_idx, train_ids, val_ids), ...]
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    splits = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(img_ids)):
        train_ids = [img_ids[i] for i in train_idx]
        val_ids = [img_ids[i] for i in val_idx]
        splits.append((fold, train_ids, val_ids))
    
    return splits


def get_image_ids(data_dir, dataset_name, img_ext='.png'):
    """
    获取数据集的所有图片ID
    
    Args:
        data_dir: 数据根目录
        dataset_name: 数据集名称
        img_ext: 图片扩展名
    
    Returns:
        list: 图片ID列表
    """
    img_paths = sorted(glob(os.path.join(data_dir, dataset_name, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
    return img_ids


def run_kfold_experiment(config, dataset_name, single_fold_fn, summarize_fn=None):
    """
    运行k折交叉验证实验
    
    Args:
        config: 配置字典
        dataset_name: 数据集名称
        single_fold_fn: 单折实验函数，接受(fold_config, exp_suffix)参数
        summarize_fn: 汇总函数，可选
    
    Returns:
        list: 所有折的实验结果
    """
    # 导入dataset_info模块（使用绝对路径，避免相对导入问题）
    current_data_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_info_path = os.path.join(current_data_dir, "dataset_info.py")
    spec = importlib.util.spec_from_file_location("data_dataset_info", dataset_info_path)
    data_dataset_info = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_dataset_info)
    get_dataset_info = data_dataset_info.get_dataset_info
    
    print(f"\n{'='*60}")
    print(f"Running {config['k_folds']}-Fold Cross Validation on {dataset_name}")
    print(f"{'='*60}")
    
    # 获取数据集信息
    img_ext, mask_ext = get_dataset_info(dataset_name)
    
    # 获取图片ID
    img_ids = get_image_ids(config['data_dir'], dataset_name, img_ext)
    print(f"Total samples: {len(img_ids)}")
    
    # 生成k折分割
    splits = get_kfold_splits(img_ids, config['k_folds'], config['dataseed'])
    
    # 确定要运行的折数
    if config['fold_to_run'] is None or config['fold_to_run'].lower() == 'all':
        folds_to_run = list(range(config['k_folds']))
    else:
        folds_to_run = [int(x.strip()) for x in config['fold_to_run'].split(',')]
    
    fold_results = []
    
    for fold, train_ids, val_ids in splits:
        if fold not in folds_to_run:
            continue
            
        print(f"\n{'-'*40}")
        print(f"Running Fold {fold+1}/{config['k_folds']}")
        print(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")
        print(f"{'-'*40}")
        
        # 为每一折创建独立的配置
        fold_config = config.copy()
        fold_config['current_fold'] = fold
        fold_config['train_ids'] = train_ids
        fold_config['val_ids'] = val_ids
        fold_config['dataset'] = dataset_name
        fold_config['img_ext'] = img_ext
        fold_config['mask_ext'] = mask_ext
        
        # 运行单折实验
        result = single_fold_fn(fold_config, f"fold{fold}")
        
        fold_results.append({
            'fold': fold,
            'dataset': dataset_name,
            'result': result
        })
    
    # 汇总k折结果（如果提供了汇总函数）
    if summarize_fn is not None:
        summarize_fn(fold_results, dataset_name, config)
    
    return fold_results

