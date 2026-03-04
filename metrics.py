import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

def iou_score_gpu(output, target, device=None):
    """GPU加速的IoU计算"""
    smooth = 1e-5
    
    if device is None:
        device = output.device if torch.is_tensor(output) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保tensors在GPU上
    if not torch.is_tensor(output):
        output = torch.tensor(output, device=device)
    if not torch.is_tensor(target):
        target = torch.tensor(target, device=device)
    
    output = torch.sigmoid(output)
    output_ = (output > 0.5).float()
    target_ = (target > 0.5).float()
    
    intersection = (output_ * target_).sum()
    union = output_.sum() + target_.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (output_.sum() + target_.sum() + smooth)
    
    # HD95需要在CPU上计算，但只在最后转换
    try:
        output_cpu = output_.cpu().numpy()
        target_cpu = target_.cpu().numpy()
        hd95_ = hausdorff_distance_95_gpu(output_cpu, target_cpu)
    except:
        hd95_ = 0
    
    return iou.item(), dice.item(), hd95_

def dice_coef_gpu(output, target, device=None):
    """GPU加速的Dice系数计算"""
    smooth = 1e-5
    
    if device is None:
        device = output.device if torch.is_tensor(output) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.is_tensor(output):
        output = torch.tensor(output, device=device)
    if not torch.is_tensor(target):
        target = torch.tensor(target, device=device)
    
    output = torch.sigmoid(output).view(-1)
    target = target.view(-1)
    
    intersection = (output * target).sum()
    
    return ((2. * intersection + smooth) / (output.sum() + target.sum() + smooth)).item()

def batch_iou_gpu(output, target, device=None):
    """批量IoU计算，适用于整个batch"""
    smooth = 1e-5
    
    if device is None:
        device = output.device if torch.is_tensor(output) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.is_tensor(output):
        output = torch.tensor(output, device=device)
    if not torch.is_tensor(target):
        target = torch.tensor(target, device=device)
    
    output = torch.sigmoid(output)
    output_ = (output > 0.5).float()
    target_ = (target > 0.5).float()
    
    # 按batch维度计算
    intersection = (output_ * target_).sum(dim=[1, 2, 3])  # 假设是4D tensor (B, C, H, W)
    union = output_.sum(dim=[1, 2, 3]) + target_.sum(dim=[1, 2, 3]) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (output_.sum(dim=[1, 2, 3]) + target_.sum(dim=[1, 2, 3]) + smooth)
    
    return iou, dice

def hausdorff_distance_95_gpu(pred, gt):
    """优化的HD95计算"""
    if pred.sum() == 0 and gt.sum() == 0:
        return 0
    if pred.sum() == 0 or gt.sum() == 0:
        return 100  # 返回一个大值表示完全不匹配
    
    # 获取边界像素
    pred_border = get_border_pixels(pred)
    gt_border = get_border_pixels(gt)
    
    if len(pred_border) == 0 or len(gt_border) == 0:
        return 100
    
    # 计算距离矩阵
    distances1 = np.array([np.min(np.linalg.norm(pred_border - gt_point, axis=1)) 
                          for gt_point in gt_border])
    distances2 = np.array([np.min(np.linalg.norm(gt_border - pred_point, axis=1)) 
                          for pred_point in pred_border])
    
    all_distances = np.concatenate([distances1, distances2])
    
    return np.percentile(all_distances, 95)

def get_border_pixels(binary_mask):
    """获取二值mask的边界像素坐标"""
    # 使用形态学操作找到边界
    from scipy import ndimage
    eroded = ndimage.binary_erosion(binary_mask)
    border = binary_mask ^ eroded
    return np.array(np.where(border)).T

def indicators_gpu(output, target, device=None):
    """GPU加速的综合指标计算"""
    if device is None:
        device = output.device if torch.is_tensor(output) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.is_tensor(output):
        output = torch.tensor(output, device=device)
    if not torch.is_tensor(target):
        target = torch.tensor(target, device=device)
    
    output = torch.sigmoid(output)
    output_ = (output > 0.5).float()
    target_ = (target > 0.5).float()
    
    # 在GPU上计算所有可以并行计算的指标
    intersection = (output_ * target_).sum()
    output_sum = output_.sum()
    target_sum = target_.sum()
    union = output_sum + target_sum - intersection
    
    # IoU
    iou_ = (intersection / union).item() if union > 0 else 0
    
    # Dice
    dice_ = (2 * intersection / (output_sum + target_sum)).item() if (output_sum + target_sum) > 0 else 0
    
    # True Positives, False Positives, False Negatives, True Negatives
    tp = intersection
    fp = output_sum - intersection
    fn = target_sum - intersection
    tn = output_.numel() - tp - fp - fn
    
    # Precision, Recall, Specificity
    precision_ = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0
    recall_ = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0
    specificity_ = (tn / (tn + fp)).item() if (tn + fp) > 0 else 0
    
    # Hausdorff距离需要在CPU上计算
    try:
        output_cpu = output_.cpu().numpy()
        target_cpu = target_.cpu().numpy()
        hd_ = hausdorff_distance_95_gpu(output_cpu, target_cpu)  # 实际是HD95
        hd95_ = hd_  # 这里简化处理
    except:
        hd_ = 0
        hd95_ = 0
    
    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_

# 使用示例和性能优化建议
def efficient_batch_evaluation(model, dataloader, device):
    """高效的批量评估函数"""
    model.eval()
    all_ious = []
    all_dices = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            # 批量计算IoU和Dice
            batch_iou, batch_dice = batch_iou_gpu(outputs, targets, device)
            
            all_ious.extend(batch_iou.cpu().numpy())
            all_dices.extend(batch_dice.cpu().numpy())
    
    return np.mean(all_ious), np.mean(all_dices)

# 内存优化版本
def memory_efficient_indicators(output, target, device=None, chunk_size=1000000):
    """内存优化的指标计算，适用于大图像"""
    if device is None:
        device = output.device if torch.is_tensor(output) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.is_tensor(output):
        output = torch.tensor(output, device=device)
    if not torch.is_tensor(target):
        target = torch.tensor(target, device=device)
    
    output = torch.sigmoid(output)
    output_ = (output > 0.5).float()
    target_ = (target > 0.5).float()
    
    # 展平并分块处理
    output_flat = output_.view(-1)
    target_flat = target_.view(-1)
    
    total_intersection = 0
    total_output_sum = 0
    total_target_sum = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    for i in range(0, len(output_flat), chunk_size):
        chunk_output = output_flat[i:i+chunk_size]
        chunk_target = target_flat[i:i+chunk_size]
        
        intersection = (chunk_output * chunk_target).sum()
        output_sum = chunk_output.sum()
        target_sum = chunk_target.sum()
        
        tp = intersection
        fp = output_sum - intersection
        fn = target_sum - intersection
        tn = len(chunk_output) - tp - fp - fn
        
        total_intersection += intersection
        total_output_sum += output_sum
        total_target_sum += target_sum
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
    
    # 计算最终指标
    union = total_output_sum + total_target_sum - total_intersection
    iou_ = (total_intersection / union).item() if union > 0 else 0
    dice_ = (2 * total_intersection / (total_output_sum + total_target_sum)).item() if (total_output_sum + total_target_sum) > 0 else 0
    precision_ = (total_tp / (total_tp + total_fp)).item() if (total_tp + total_fp) > 0 else 0
    recall_ = (total_tp / (total_tp + total_fn)).item() if (total_tp + total_fn) > 0 else 0
    specificity_ = (total_tn / (total_tn + total_fp)).item() if (total_tn + total_fp) > 0 else 0
    
    return iou_, dice_, precision_, recall_, specificity_