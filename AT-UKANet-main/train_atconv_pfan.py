"""
AT-UKanNet 训练脚本（UKAN + ATConv + 并行 CASA）

支持消融实验：
+1: 使用 ATConv（encoder 若干层, decoder 若干层）替代 UKAN 中对应的标准卷积
+2: 在特征提取阶段引入并行 CASA 注意力（Channel-wise Attention + Spatial Attention，并行结构，基于并行 CBAM 实现）
+3: 在网络架构层面结合注意力（例如在 decoder1 / decoder2 之后加入注意力模块，当前默认关闭）
+4: 在损失函数层面引入 PFAN 风格的 Edge Preservation Loss（边界保持损失，已验证提升有限，默认关闭）

说明：
- PFAN 与 ARConv 是项目早期尝试的改进模块，实验效果一般，现作为历史对比方案保留，不再作为推荐配置。
- 当前推荐配置为：use_atconv=True，use_attention=True，attention_variant='parallel'（并行 CASA）。
支持 5 折交叉验证。
"""
import os
import sys
from datetime import datetime

# 添加当前目录和父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 确保当前目录优先（避免父目录的utils.py干扰）
# 将当前目录插入到最前面
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# 父目录放在后面
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 导入模块化组件（使用当前目录的包）
# 由于父目录有utils.py，需要使用明确的文件路径导入
import importlib.util

def import_module_from_path(module_name, file_path):
    """从文件路径导入模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入utils模块
utils_config = import_module_from_path("utils_config", os.path.join(current_dir, "utils", "config.py"))
parse_args = utils_config.parse_args

utils_seed = import_module_from_path("utils_seed", os.path.join(current_dir, "utils", "seed.py"))
seed_torch = utils_seed.seed_torch

# 导入data模块
data_kfold = import_module_from_path("data_kfold", os.path.join(current_dir, "data", "kfold.py"))
run_kfold_experiment = data_kfold.run_kfold_experiment

# 导入train模块
train_kfold_trainer = import_module_from_path("train_kfold_trainer", os.path.join(current_dir, "train", "kfold_trainer.py"))
run_single_fold_experiment = train_kfold_trainer.run_single_fold_experiment

train_metrics = import_module_from_path("train_metrics", os.path.join(current_dir, "train", "metrics.py"))
summarize_kfold_results = train_metrics.summarize_kfold_results


def main():
    """主函数"""
    # 设置随机种子
    seed_torch()
    
    # 解析参数
    config = vars(parse_args())

    # 生成实验名称
    exp_name = config.get('name')
    output_dir = config.get('output_dir')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if exp_name is None:
        # 根据消融实验配置生成实验名称
        ablation_suffix = []
        if config.get('use_atconv') and config.get('atconv_encoder_layers') != 'none':
            ablation_suffix.append('+1')
        if config.get('use_attention'):
            ablation_suffix.append('+2')
        if config.get('use_hybrid_arch'):
            ablation_suffix.append('+3')
        if config.get('use_edge_loss'):
            ablation_suffix.append('+4')
        
        if ablation_suffix:
            ablation_str = '_'.join(ablation_suffix)
        else:
            ablation_str = 'baseline'
        
        config['name'] = f"{timestamp}_{config['datasets']}_{config['arch']}_{ablation_str}"
    else:
        config['name'] = f"{timestamp}_{exp_name}"

    os.makedirs(output_dir, exist_ok=True)

    # 打印配置信息（对应 AT-UKanNet 的四个消融开关）
    print('-' * 20)
    print('AT-UKanNet Ablation Study Configuration:')
    print(f'  +1 (ATConv): {config.get("use_atconv", False)} (encoder: {config.get("atconv_encoder_layers", "3")}, decoder: {config.get("atconv_decoder_layers", "none")})')
    print(f'  +2 (Attention): {config.get("use_attention", False)}')
    if config.get("use_attention", False): # 如果 use_attention 为 True，则执行 print 语句。
        print(f'      Variant: {config.get("attention_variant", "serial")}') # 打印 attention_variant 的值，如果为 None，则打印 "serial"。      
    print(f'  +3 (Hybrid Arch): {config.get("use_hybrid_arch", False)}')
    print(f'  +4 (Edge Loss): {config.get("use_edge_loss", False)}')
    print('-' * 20)

    # 运行k折交叉验证
    fold_results = run_kfold_experiment(
        config, 
        config['datasets'], 
        run_single_fold_experiment,
        summarize_kfold_results
    )


if __name__ == '__main__':
    main()

