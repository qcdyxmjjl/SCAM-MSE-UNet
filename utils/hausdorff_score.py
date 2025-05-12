import numpy as np
from scipy.ndimage import distance_transform_edt

def hausdorff_95(pred, target):
    """
    Calculate the 95th percentile of the Hausdorff Distance
    """
    # 确保转换为numpy数组并且是布尔类型
    pred = pred.cpu().numpy().astype(bool)
    target = target.cpu().numpy().astype(bool)
    
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    elif pred.sum() == 0 or target.sum() == 0:
        return np.inf
        
    # 计算两个方向的距离
    d_map1 = distance_transform_edt(~pred)
    d_map2 = distance_transform_edt(~target)
    
    # 获取边界点的距离
    pred_boundary = d_map2[pred]
    target_boundary = d_map1[target]
    
    if len(pred_boundary) == 0 or len(target_boundary) == 0:
        return np.inf
        
    # 计算双向Hausdorff距离
    h1 = np.percentile(pred_boundary, 95)
    h2 = np.percentile(target_boundary, 95)
    
    return max(h1, h2) 