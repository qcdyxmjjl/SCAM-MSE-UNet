import torch

def voe_score(pred, target):
    """
    Calculate Volumetric Overlap Error (VOE)
    VOE = 1 - |A∩B|/(|A|+|B|-|A∩B|)
    """
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    
    if union == 0:
        return 0.0
        
    voe = 1.0 - (intersection.float() / union.float())
    return voe.item() 