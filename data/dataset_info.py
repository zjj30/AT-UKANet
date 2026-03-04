"""
数据集信息模块
支持7个数据集：busi, glas, cvc, ours, busi_whu, TN3K, BUS-BRA
"""
def get_dataset_info(dataset_name):
    """
    获取数据集的文件扩展名信息
    
    Args:
        dataset_name: 数据集名称
            原始4个：busi, glas, cvc, ours
            新增3个：busi_whu, TN3K, BUS-BRA
    
    Returns:
        tuple: (img_ext, mask_ext)
    """
    # 默认图片扩展名
    img_ext = '.png'
    
    # 原始4个数据集
    if dataset_name == 'busi':
        mask_ext = '_mask.png'  # busi数据集使用特殊命名：image_mask.png
    elif dataset_name == 'glas':
        mask_ext = '.png'  # 标准格式：image.png 和 mask.png
    elif dataset_name == 'cvc':
        mask_ext = '.png'  # 标准格式
    elif dataset_name == 'ours':
        mask_ext = '.png'  # 标准格式
    # 新增3个数据集
    elif dataset_name == 'busi_whu':
        img_ext = '.bmp'  # busi_whu图片使用.bmp格式
        mask_ext = '_anno.bmp'  # mask文件命名格式：{img_id}_anno.bmp
    elif dataset_name == 'TN3K':
        img_ext = '.jpg'  # TN3K图片使用.jpg格式
        mask_ext = '.jpg'  # mask使用标准格式
    elif dataset_name == 'BUS-BRA' or dataset_name == 'BUS_BRA' or dataset_name == 'bus_bra' or dataset_name == 'bus_bar':
        # 支持多种命名方式（连字符、下划线、小写、可能的拼写变体）
        img_ext = '.png'  # bus_bar图片使用.png格式
        mask_ext = 'mask_.png'  # mask文件命名格式：mask_{img_id}.png（注意：Dataset类会处理前缀）
    else:
        # 默认使用标准格式
        mask_ext = '.png'
        print(f"警告: 未知数据集 '{dataset_name}'，使用默认mask扩展名 '.png'")
    
    return img_ext, mask_ext

