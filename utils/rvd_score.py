import torch

def rvd_score(pred, target):
    """
    Calculate Relative Volume Difference (RVD)
    RVD = (|A|-|B|)/|B| * 100%
    """
    pred_volume = pred.sum()
    target_volume = target.sum()
    
    if target_volume == 0:
        return 0.0
        
    rvd = ((pred_volume - target_volume).float() / target_volume.float()) * 100
    return rvd.item() 