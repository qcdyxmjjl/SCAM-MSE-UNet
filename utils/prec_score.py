import torch
from torch import Tensor


def precision_score(pred, target):
    """
    Calculate Precision score
    Precision = TP/(TP+FP)
    """
    true_positives = torch.logical_and(pred, target).sum()
    predicted_positives = pred.sum()
    
    if predicted_positives == 0:
        return 0.0
        
    precision = true_positives.float() / predicted_positives.float()
    return precision.item()