"""
AT-UKanNet 可视化预测脚本
使用 AT-UKanNet 模型（UKAN + ATConv + 并行 CASA 注意力）进行预测并保存 mask。

参数配置：
- 数据集: glas, busi, bus_bra, ours, busi_whu, TN3K, cvc
- 模型目录: 通过 --model_dir 指定，或自动从数据集查找
- 图片 ID: 通过 --img_ids 指定，或使用数据集默认值

说明：
- 早期实验目录中仍可能使用 `UKAN_ATConv_PFAN` 等旧命名，本脚本依然兼容这些目录结构，
  但模型实现本身已经升级为 AT-UKanNet（类名 `AT_UKanNet`，兼容别名 `UKAN_PFAN`）。
"""

import os
import sys
import argparse
import yaml
from glob import glob
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 导入筛选和可视化模块
try:
    import sys
    import os
    # 获取 AT-UKanNet（历史 UKAN_ATConv_PFAN）目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from filter_and_visualize import (
        calculate_dice,
        create_difference_map,
        overlay_difference_map_on_image,
        add_dice_annotation
    )
    HAS_FILTER_VIS = True
except ImportError as e:
    HAS_FILTER_VIS = False
    print(f"警告: 无法导入filter_and_visualize模块，将跳过差异热力图和Dice标注功能: {e}")

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加 AT-UKanNet（UKAN_ATConv_PFAN）实验目录到路径（兼容历史路径命名）
ukan_atconv_path = os.path.join(project_root, "UKAN_ATConv_PFAN", "experiments_cbam_variants")
if ukan_atconv_path not in sys.path:
    sys.path.insert(0, ukan_atconv_path)

# 导入必要的模块
from dataset import Dataset
from nets.archs_ukan_pfan import UKAN_PFAN


from albumentations import Compose, Resize
from albumentations.augmentations import transforms
HAS_ALBUMENTATIONS = True


class SegmentationGradCAM:
    """
    针对分割模型的简单 Grad-CAM/Grad-CAM++ 实现。
    默认对最后一层特征（例如 decoder 最后一个卷积块）做 CAM。
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_handle = target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    def __del__(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate_cam(self, input_tensor: torch.Tensor, target_mask: torch.Tensor = None) -> np.ndarray:
        """
        input_tensor: (1, C, H, W)
        target_mask: (1, 1, H, W) 或 None；若提供，则以 mask 内区域平均 logit 作为目标
        """
        self.model.zero_grad()

        # 处理 deep supervision：获取最终输出
        outputs = self.model(input_tensor)
        if isinstance(outputs, (list, tuple)):
            output = outputs[-1]  # 使用最终输出
        else:
            output = outputs

        # 确保 output 是正确的形状
        if output.dim() == 4 and output.shape[1] == 1:
            output = output  # (1, 1, H, W)
        elif output.dim() == 3:
            output = output.unsqueeze(1)  # 添加通道维度

        if target_mask is not None:
            # 只对前景区域的 logit 求平均，作为目标
            score = (output * target_mask).sum() / (target_mask.sum() + 1e-6)
        else:
            # 全图平均
            score = output.mean()

        score.backward(retain_graph=True)

        # 检查梯度是否被捕获
        if self.gradients is None:
            raise ValueError("未捕获到梯度，请检查target layer选择是否正确")

        # Grad-CAM 权重: 对空间位置求平均
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        if activations is None:
            raise ValueError("未捕获到激活值，请检查target layer选择是否正确")

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)
        cam = cam[0, 0].cpu().numpy()

        # 归一化到 0-1
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_cam_on_image(img: np.ndarray, cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    img: (H,W,3) 原图（0-255）
    cam: (h,w) 0-1 CAM，需插值到 H,W
    """
    H, W, _ = img.shape
    cam_resized = cv2.resize(cam, (W, H))
    heatmap_bgr = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = (0.4 * heatmap + 0.6 * img).astype(np.uint8)
    return heatmap, overlay


def get_target_layer_for_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    为UKAN模型选择合适的target layer用于Grad-CAM。
    选择输出特征图尺寸较大的层，确保生成的CAM有足够的空间分辨率。
    """
    # 对于UKAN模型，尝试找到输出特征图较大的层

    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((name, module))

    if conv_layers:
        # 选择策略：优先选择名称中包含'stage'或'level'且数字较大的层
        # 这些通常是较高分辨率的特征层
        scored_layers = []
        for i, (name, module) in enumerate(conv_layers):
            score = 0
            name_lower = name.lower()

            # 优先级1: decoder相关层
            if 'decoder' in name_lower or 'up' in name_lower:
                score += 100

            # 优先级2: stage/level编号较大的层
            import re
            stage_match = re.search(r'stage(\d+)', name_lower)
            level_match = re.search(r'level(\d+)', name_lower)
            if stage_match:
                score += int(stage_match.group(1)) * 10
            if level_match:
                score += int(level_match.group(1)) * 10

            # 优先级3: 位置越靠后得分越高（通常分辨率越低但特征越丰富）
            score += i * 0.1

            scored_layers.append((score, name, module))

        # 按得分排序，选择得分最高的层，但要确保不是输出层（最后一个）
        scored_layers.sort(key=lambda x: x[0], reverse=True)

        # 避免选择最后一个卷积层（通常是输出层，特征图可能很小）
        if len(scored_layers) > 1:
            selected_layer = scored_layers[0] if scored_layers[0][1] != conv_layers[-1][0] else scored_layers[1]
        else:
            selected_layer = scored_layers[0]

        selected_name, selected_module = selected_layer[1], selected_layer[2]

        print(f"    选择Grad-CAM目标层: {selected_name} (得分: {selected_layer[0]:.1f})")
        return selected_module

    # 回退到原来的逻辑
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            return module

    # 最后的回退
    all_modules = list(model.modules())
    return all_modules[-1]


# 从环境变量获取数据目录，如果未设置则使用默认值
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "/mnt/a/weather-ZJJ/ZhuJJ/UKan/inputs")

# 根据数据集查找可用模型的函数 (移植自shell脚本)
def find_available_model(dataset: str) -> str:
    """
    根据数据集名称查找可用的模型目录

    Args:
        dataset: 数据集名称

    Returns:
        model_dir: 模型目录路径 (相对于experiments_cbam_variants目录)
    """
    model_dir = ""

    if dataset == "glas":
        model_dir = "outputs/20260110_042514_UKAN_ATConv_glas_seed8142_fold0"
    elif dataset == "busi":
        model_dir = "outputs/20260110_064118_UKAN_ATConv_busi_seed2981_fold3"
    elif dataset == "bus_bra":
        model_dir = "outputs/20260120_121157_UKAN+1+2_parallel_fold3_gpu3_p1/fold3"
    elif dataset == "ours":
        model_dir = "outputs/20260111_180748_UKAN_ATConv_ours_seed2224/fold3"
    elif dataset == "busi_whu":
        model_dir = "outputs/20260113_035358_UKAN_ATConv_busi_whu_seed1234/fold0"
    elif dataset == "TN3K":
        model_dir = "outputs/20260115_183059_UKAN_ATConv_TN3K_seed5678/fold2"
    elif dataset == "cvc":
        model_dir = "outputs/20260115_011454_UKAN+1+2_parallel_fold4_gpu0_p1/fold4"

    return model_dir

# 获取数据集默认配置的函数 (移植自shell脚本)
def get_default_img_ids(dataset: str) -> str:
    """
    根据数据集名称获取默认的图片ID列表

    Args:
        dataset: 数据集名称

    Returns:
        default_img_ids: 默认图片ID列表 (逗号分隔的字符串)
    """
    default_img_ids = ""

    if dataset == "glas":
        default_img_ids = "5,18,134"
    elif dataset == "busi":
        default_img_ids = "benign (1),benign (45),malignant (15)"
    elif dataset == "bus_bra":
        default_img_ids = "bus_0001-l,bus_0002-r,bus_0003-l,bus_0259-l,bus_0303-l"
    elif dataset == "ours":
        default_img_ids = "20210119160316941,20210301112200510,20211109082507190"
    elif dataset == "busi_whu":
        default_img_ids = "00009,00567,10070-3"
    elif dataset == "TN3K":
        default_img_ids = "0259,0508,1120"
    elif dataset == "cvc":
        default_img_ids = "5,18,134"

    return default_img_ids

def load_config_and_model(exp_dir: str, device: str = "cuda"):
    """
    从实验目录加载config.yml和模型权重

    Args:
        exp_dir: 实验目录路径
        device: 设备

    Returns:
        config: 配置字典
        model: 加载的模型
    """
    cfg_path = os.path.join(exp_dir, "config.yml")

    # 尝试多种模型文件命名方式
    possible_model_files = ["model.pth", "model_best_iou.pth", "model_best_f1.pth", "model_best.pth"]
    ckpt_path = None
    for model_file in possible_model_files:
        potential_path = os.path.join(exp_dir, model_file)
        if os.path.exists(potential_path):
            ckpt_path = potential_path
            break

    if ckpt_path is None:
        raise FileNotFoundError(f"在目录 {exp_dir} 中找不到模型权重文件。尝试的文件: {possible_model_files}")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    # 读取配置文件
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(f"加载配置文件: {cfg_path}")
    print(f"加载模型权重: {ckpt_path}")

    # 创建 AT-UKanNet 模型（兼容旧别名 UKAN_PFAN）
    model = UKAN_PFAN(
        num_classes=config["num_classes"],
        input_channels=config["input_channels"],
        deep_supervision=config["deep_supervision"],
        embed_dims=config["input_list"],
        no_kan=config.get("no_kan", False),
        # AT-UKanNet 特定参数
        use_atconv=config.get("use_atconv", True),
        atconv_encoder_layers=config.get("atconv_encoder_layers", "3"),
        atconv_decoder_layers=config.get("atconv_decoder_layers", "1"),
        use_attention=config.get("use_attention", True),
        attention_variant=config.get("attention_variant", "parallel"),
        use_hybrid_arch=config.get("use_hybrid_arch", False),
    )

    # 加载模型权重
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state)
        print("模型权重加载成功")
    except Exception as e:
        print(f"直接加载权重失败，尝试strict=False: {e}")
        model.load_state_dict(state, strict=False)
        print("模型权重加载成功 (strict=False)")

    model.to(device)
    model.eval()
    return config, model

def build_val_dataset(config: Dict, img_ids: List[str], data_dir: str):
    """
    根据配置构建验证数据集

    Args:
        config: 配置字典
        img_ids: 图片ID列表
        data_dir: 数据目录

    Returns:
        dataset: Dataset对象
    """
    dataset_name = config["dataset"]

    # 根据数据集类型设置mask扩展名
    mask_ext_map = {
        "bus_bar": "mask_.png",
        "busi_whu": "_anno.bmp",
        "TN3K": ".jpg",
        "busi": "_mask.png"
    }
    mask_ext = mask_ext_map.get(dataset_name, ".png")

    # 根据数据集类型设置img扩展名
    img_ext_map = {
        "bus_bra": ".png",
        "busi_whu": ".bmp",
        "TN3K": ".jpg",
        "busi": ".png"
    }
    img_ext = img_ext_map.get(dataset_name, ".png")

    img_dir = os.path.join(data_dir, dataset_name, "images")
    mask_dir = os.path.join(data_dir, dataset_name, "masks")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")

    if HAS_ALBUMENTATIONS:
        val_transform = Compose([
            Resize(config["input_h"], config["input_w"]),
            transforms.Normalize(),
        ])
    else:
        # 使用简单的变换
        val_transform = lambda x: cv2.resize(x, (config["input_w"], config["input_h"]))

    dataset = Dataset(
        img_ids=img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config["num_classes"],
        transform=val_transform,
    )
    return dataset

def predict_and_save_masks(dataset_name: str, model_dir: str, img_ids: List[str], output_dir: str, device: str = "cuda", generate_gradcam: bool = False):
    """
    对指定数据集进行预测并保存mask

    Args:
        dataset_name: 数据集名称
        model_dir: 模型目录路径
        img_ids: 图片ID列表
        output_dir: 输出目录
        device: 设备
    """
    print(f"\n🔄 开始处理数据集: {dataset_name}")
    print(f"📁 数据目录: {DEFAULT_DATA_DIR}")
    print(f"📂 实验目录: {model_dir}")
    print(f"🖼️  测试图片: {img_ids}")

    try:
        # 加载配置和模型
        config, model = load_config_and_model(model_dir, device)
        print("✅ 模型加载成功")

        # 初始化Grad-CAM helper（如果需要）
        cam_helper = None
        if generate_gradcam:
            target_layer = get_target_layer_for_model(model)
            cam_helper = SegmentationGradCAM(model, target_layer)
            print(f"  ✅ Grad-CAM初始化成功，使用目标层: {target_layer.__class__.__name__}")

        # 创建数据集
        dataset = build_val_dataset(config, img_ids, DEFAULT_DATA_DIR)
        print(f"✅ 数据集创建成功，包含 {len(dataset)} 张图片")

        # 创建输出目录
        model_output_dir = os.path.join(output_dir, f"AT_UKanNet_{dataset_name}")
        os.makedirs(model_output_dir, exist_ok=True)

        # 创建Grad-CAM输出目录
        gradcam_output_dir = None
        if generate_gradcam:
            gradcam_output_dir = os.path.join(output_dir, f"AT_UKanNet_{dataset_name}_gradcam")
            os.makedirs(gradcam_output_dir, exist_ok=True)
            print(f"  📁 Grad-CAM输出目录: {gradcam_output_dir}")

        print(f"📁 输出目录: {model_output_dir}")

        # 对每张图片进行预测
        for idx in range(len(dataset)):
            img, gt_mask, meta = dataset[idx]
            img_id = meta["img_id"]

            # 准备输入tensor（用于预测和Grad-CAM）
            inp = torch.from_numpy(img).unsqueeze(0).to(device)  # (1,C,H,W)

            with torch.no_grad():
                out = model(inp)
                out = torch.sigmoid(out).cpu().numpy()[0, 0]  # 取第一个通道
                pred_mask = (out >= 0.5).astype(np.uint8) * 255

            # 保存原始图像
            # 根据数据集确定正确的文件扩展名
            img_ext_map = {
                "bus_bra": ".png",
                "busi_whu": ".bmp",
                "TN3K": ".jpg",
                "busi": ".png"
            }
            img_ext = img_ext_map.get(dataset_name, ".png")

            # 处理数据集名称到目录名称的映射
            path_dataset_name = "bus_bar" if dataset_name == "bus_bra" else dataset_name
            src_img_path = os.path.join(DEFAULT_DATA_DIR, path_dataset_name, "images", f"{img_id}{img_ext}")

            if os.path.exists(src_img_path):
                # 读取原始图像并resize到模型输入尺寸
                original_img = cv2.imread(src_img_path)
                if original_img is not None:
                    # resize到模型输入尺寸以保持一致性
                    target_size = (config["input_w"], config["input_h"])
                    resized_img = cv2.resize(original_img, target_size, interpolation=cv2.INTER_LINEAR)
                    dst_img_path = os.path.join(model_output_dir, f"{img_id}_image.png")
                    cv2.imwrite(dst_img_path, resized_img)
                    print(f"    🖼️  保存resize后原图: {dst_img_path}")
                else:
                    print(f"    ⚠️  无法读取原始图像文件: {src_img_path}")
                # 备用方案：保存处理后的图像（反归一化）
                fallback_img_save(img, model_output_dir, img_id, gt_mask, pred_mask)
            else:
                print(f"    ⚠️  原始图像文件不存在: {src_img_path}")
                # 备用方案：保存处理后的图像（反归一化）
                fallback_img_save(img, model_output_dir, img_id, gt_mask, pred_mask)

            # 保存ground truth mask（如果存在）
            if gt_mask is not None:
                gt_save_path = os.path.join(model_output_dir, f"{img_id}_gt.png")
                # 简化形状处理：移除多余的通道维度
                if gt_mask.ndim == 3:
                    gt_mask_processed = gt_mask.squeeze(axis=0 if gt_mask.shape[0] == 1 else -1)
                else:
                    gt_mask_processed = gt_mask
                gt_mask_uint8 = (gt_mask_processed * 255).astype(np.uint8)
                cv2.imwrite(gt_save_path, gt_mask_uint8)
                print(f"    🎯 保存GT: {gt_save_path}")

            # 保存预测mask
            pred_save_path = os.path.join(model_output_dir, f"{img_id}_pred.png")
            cv2.imwrite(pred_save_path, pred_mask)
            print(f"    🤖 保存预测: {pred_save_path}")
            
            # 如果启用了筛选和可视化功能，生成差异热力图和Dice标注
            if HAS_FILTER_VIS and gt_mask is not None:
                try:
                    gt_mask_uint8 = (gt_mask * 255).astype(np.uint8)
                    if gt_mask_uint8.ndim == 3:
                        gt_mask_uint8 = gt_mask_uint8.squeeze(axis=0 if gt_mask_uint8.shape[0] == 1 else -1)
                    
                    # 计算Dice分数
                    dice = calculate_dice(pred_mask, gt_mask_uint8)
                    
                    # 读取原图用于可视化
                    if os.path.exists(src_img_path):
                        vis_img = cv2.imread(src_img_path)
                        if vis_img is not None:
                            # resize到与mask相同的尺寸
                            pred_mask_h, pred_mask_w = pred_mask.shape
                            if vis_img.shape[:2] != (pred_mask_h, pred_mask_w):
                                vis_img = cv2.resize(vis_img, (pred_mask_w, pred_mask_h), interpolation=cv2.INTER_LINEAR)
                            
                            # 创建差异热力图
                            diff_map = create_difference_map(pred_mask, gt_mask_uint8, size=(128, 128))
                            
                            # 叠加差异热力图到原图
                            vis_img_with_diff = overlay_difference_map_on_image(vis_img, diff_map, position="bottom_right", scale=0.25)
                            
                            # 添加Dice标注
                            vis_img_with_diff = add_dice_annotation(vis_img_with_diff, dice, position="bottom")
                            
                            # 保存可视化结果
                            vis_save_path = os.path.join(model_output_dir, f"{img_id}_vis.png")
                            cv2.imwrite(vis_save_path, vis_img_with_diff)
                            print(f"    🎨 保存可视化: {vis_save_path} (Dice: {dice:.3f})")
                            
                            # 保存差异热力图
                            diff_save_path = os.path.join(model_output_dir, f"{img_id}_diff.png")
                            cv2.imwrite(diff_save_path, diff_map)
                except Exception as e:
                    print(f"    ⚠️  可视化生成失败: {e}")
                    import traceback
                    traceback.print_exc()

            # 生成和保存Grad-CAM热力图（如果启用）
            if generate_gradcam and cam_helper is not None and gradcam_output_dir is not None:
                try:
                    # 准备目标mask用于Grad-CAM
                    if gt_mask is not None:
                        tgt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(device)
                    else:
                        tgt_mask = None

                    # 确保输入tensor需要梯度（用于Grad-CAM）
                    inp.requires_grad_(True)

                    # 生成CAM
                    cam = cam_helper.generate_cam(inp, tgt_mask)
                    print(f"    Grad-CAM尺寸: {cam.shape}, 范围: [{cam.min():.3f}, {cam.max():.3f}]")

                    # 获取原始图像用于可视化
                    if os.path.exists(src_img_path):
                        original_img = cv2.imread(src_img_path)
                        if original_img is not None:
                            # resize原始图像到网络输入尺寸（与预测时一致）
                            _, _, H_in, W_in = inp.shape
                            if original_img.shape[:2] != (H_in, W_in):
                                original_img = cv2.resize(original_img, (W_in, H_in), interpolation=cv2.INTER_LINEAR)

                            # 生成热力图和叠加图（CAM会自动resize到original_img的尺寸）
                            heatmap, overlay = overlay_cam_on_image(original_img, cam)

                            # 保存Grad-CAM相关文件
                            # 保存原图
                            gradcam_img_path = os.path.join(gradcam_output_dir, f"{img_id}_original.png")
                            cv2.imwrite(gradcam_img_path, original_img)
                            print(f"    🎨 保存Grad-CAM原图: {gradcam_img_path}")

                            # 保存热力图
                            gradcam_heatmap_path = os.path.join(gradcam_output_dir, f"{img_id}_heatmap.png")
                            cv2.imwrite(gradcam_heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
                            print(f"    🎨 保存Grad-CAM热力图: {gradcam_heatmap_path}")

                            # 保存叠加图
                            gradcam_overlay_path = os.path.join(gradcam_output_dir, f"{img_id}_overlay.png")
                            cv2.imwrite(gradcam_overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                            print(f"    🎨 保存Grad-CAM叠加图: {gradcam_overlay_path}")

                            # 保存GT mask（如果存在）
                            if gt_mask is not None:
                                gt_mask_uint8 = (gt_mask * 255).astype(np.uint8)
                                # 简化形状处理：移除多余的通道维度
                                if gt_mask_uint8.ndim == 3:
                                    gt_mask_uint8 = gt_mask_uint8.squeeze(axis=0 if gt_mask_uint8.shape[0] == 1 else -1)

                                gradcam_gt_path = os.path.join(gradcam_output_dir, f"{img_id}_gt.png")
                                cv2.imwrite(gradcam_gt_path, gt_mask_uint8)
                                print(f"    🎨 保存Grad-CAM GT: {gradcam_gt_path}")

                        else:
                            print(f"    ❌ Grad-CAM: 读取原图失败: {src_img_path}")
                    else:
                        print(f"    ❌ Grad-CAM: 原图文件不存在: {src_img_path}")

                except Exception as e:
                    print(f"    ❌ Grad-CAM生成失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        print(f"✅ 数据集 {dataset_name} 处理完成")

    except Exception as e:
        print(f"❌ 处理数据集 {dataset_name} 时出错: {e}")
        import traceback
        traceback.print_exc()


def fallback_img_save(img, model_output_dir, img_id, gt_mask, pred_mask):
    """备用方案：保存处理后的图像"""
    if img.shape[0] == 1:
        original_img = img[0]
        original_img = np.clip(original_img * 255, 0, 255).astype(np.uint8)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = img.transpose(1, 2, 0)
        original_img = (original_img * std + mean) * 255
        original_img = np.clip(original_img, 0, 255).astype(np.uint8)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    dst_img_path = os.path.join(model_output_dir, f"{img_id}_image.png")
    cv2.imwrite(dst_img_path, original_img)
    print(f"    🖼️  保存处理后原图: {dst_img_path}")

    # 保存ground truth mask
    if gt_mask is not None:
        gt_save_path = os.path.join(model_output_dir, f"{img_id}_gt.png")
        # 简化形状处理：移除多余的通道维度
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.squeeze(axis=0 if gt_mask.shape[0] == 1 else -1)
        gt_mask_uint8 = (gt_mask * 255).astype(np.uint8)
        cv2.imwrite(gt_save_path, gt_mask_uint8)
        print(f"    🎯 保存GT: {gt_save_path}")

    # 保存预测mask
    pred_save_path = os.path.join(model_output_dir, f"{img_id}_pred.png")
    cv2.imwrite(pred_save_path, pred_mask)
    print(f"    🤖 保存预测: {pred_save_path}")


def main():
    parser = argparse.ArgumentParser(description="AT-UKanNet 预测并保存 mask")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称 (glas|busi|bus_bra|ours|busi_whu|TN3K|cvc)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="模型目录路径 (相对于experiments_cbam_variants目录，如果不指定则自动查找)"
    )
    parser.add_argument(
        "--img_ids",
        type=str,
        help="图片ID列表，用逗号分隔 (如果不指定则使用默认图片ID)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predicted_masks",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备: cuda 或 cpu"
    )
    parser.add_argument(
        "--generate_gradcam",
        action="store_true",
        help="生成Grad-CAM注意力热力图",
    )

    args = parser.parse_args()

    # 获取模型目录
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = find_available_model(args.dataset)
        if not model_dir:
            raise ValueError(f"未找到数据集 {args.dataset} 的可用UKAN模型")

    # 确保模型目录是完整路径
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(os.path.dirname(__file__), model_dir)

    # 获取图片ID列表
    if args.img_ids:
        img_ids = [img_id.strip() for img_id in args.img_ids.split(',')]
    else:
        default_img_ids_str = get_default_img_ids(args.dataset)
        if not default_img_ids_str:
            raise ValueError(f"未找到数据集 {args.dataset} 的默认图片ID")
        img_ids = [img_id.strip() for img_id in default_img_ids_str.split(',')]

    print("🚀 AT-UKanNet 预测脚本")
    print("=" * 50)
    print(f"数据集: {args.dataset}")
    print(f"模型目录: {model_dir}")
    print(f"图片ID: {img_ids}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")

    predict_and_save_masks(args.dataset, model_dir, img_ids, args.output_dir, args.device, args.generate_gradcam)

    print("\n🎉 预测完成！")
    print(f"📁 结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()