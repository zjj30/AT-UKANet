"""
指标计算和汇总模块
"""
import numpy as np
import pandas as pd
import os




def summarize_kfold_results(fold_results, dataset_name, config):
    """
    汇总k折结果
    包含每个fold的详细结果（用于画箱线图）和统计汇总
    """
    if len(fold_results) == 0:
        return
    
    print(f"\n{'-'*60}")
    print(f"K-Fold Cross Validation Summary - {dataset_name}")
    print(f"{'-'*60}")
    
    exp_dir = os.path.join(config['output_dir'], config['name'])
    os.makedirs(exp_dir, exist_ok=True)
    
    # 提取每个fold的详细结果
    detailed_data = []
    all_best_iou = []
    all_best_dice = []
    all_final_iou = []
    all_final_dice = []
    
    for fold_result in fold_results:
        fold = fold_result['fold']
        result = fold_result.get('result', {})
        
        # 调试信息：打印result的结构
        if not isinstance(result, dict):
            print(f"警告: Fold {fold} 的result不是字典类型: {type(result)}")
            metrics = {}
        elif 'metrics' in result:
            metrics = result['metrics']
            if not isinstance(metrics, dict):
                print(f"警告: Fold {fold} 的metrics不是字典类型: {type(metrics)}")
                metrics = {}
        else:
            # 向后兼容：直接使用result作为metrics
            metrics = result
        
        best_iou = metrics.get('best_iou') if isinstance(metrics, dict) else None
        best_dice = metrics.get('best_dice') if isinstance(metrics, dict) else None
        final_iou = metrics.get('final_iou') if isinstance(metrics, dict) else None
        final_dice = metrics.get('final_dice') if isinstance(metrics, dict) else None
        
        # 调试信息：如果所有指标都是None，打印警告
        if all(x is None for x in [best_iou, best_dice, final_iou, final_dice]):
            print(f"警告: Fold {fold} 的所有指标都是None")
            print(f"  result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            if isinstance(result, dict) and 'metrics' in result:
                print(f"  metrics keys: {list(metrics.keys()) if isinstance(metrics, dict) else 'N/A'}")
        
        detailed_data.append({
            'fold': fold,
            'best_iou': best_iou,
            'best_dice': best_dice,
            'final_iou': final_iou,
            'final_dice': final_dice,
        })
        
        if best_iou is not None:
            all_best_iou.append(best_iou)
        if best_dice is not None:
            all_best_dice.append(best_dice)
        if final_iou is not None:
            all_final_iou.append(final_iou)
        if final_dice is not None:
            all_final_dice.append(final_dice)
    
    # 创建详细结果数据框
    detailed_df = pd.DataFrame(detailed_data)
    detailed_file = os.path.join(exp_dir, 'kfold_detailed.csv')
    detailed_df.to_csv(detailed_file, index=False)
    
    # 创建统计汇总数据框
    summary_data = []
    if len(all_best_iou) > 0:
        summary_data.append({
            'metric': 'best_iou',
            'mean': np.mean(all_best_iou),
            'std': np.std(all_best_iou),
            'min': np.min(all_best_iou),
            'max': np.max(all_best_iou),
            'median': np.median(all_best_iou),
        })
    if len(all_best_dice) > 0:
        summary_data.append({
            'metric': 'best_dice',
            'mean': np.mean(all_best_dice),
            'std': np.std(all_best_dice),
            'min': np.min(all_best_dice),
            'max': np.max(all_best_dice),
            'median': np.median(all_best_dice),
        })
    if len(all_final_iou) > 0:
        summary_data.append({
            'metric': 'final_iou',
            'mean': np.mean(all_final_iou),
            'std': np.std(all_final_iou),
            'min': np.min(all_final_iou),
            'max': np.max(all_final_iou),
            'median': np.median(all_final_iou),
        })
    if len(all_final_dice) > 0:
        summary_data.append({
            'metric': 'final_dice',
            'mean': np.mean(all_final_dice),
            'std': np.std(all_final_dice),
            'min': np.min(all_final_dice),
            'max': np.max(all_final_dice),
            'median': np.median(all_final_dice),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(exp_dir, 'kfold_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    # 打印统计信息
    print("\nDetailed Results (each fold):")
    print(detailed_df.to_string(index=False))
    
    print("\nStatistical Summary:")
    print(summary_df.to_string(index=False))
    
    print(f"\nFiles saved:")
    print(f"  - Detailed results: {detailed_file}")
    print(f"  - Summary: {summary_file}")
    print(f"{'-'*60}")