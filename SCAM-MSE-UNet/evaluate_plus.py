import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.iou_score import iou_coeff as iou_score
from utils.prec_score import precision_score
from utils.voe_score import voe_score
from utils.rvd_score import rvd_score
from utils.hausdorff_score import hausdorff_95


@torch.inference_mode()
def evaluate_plus(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_value = 0
    prec_value = 0
    acc_value = 0
    voe_value = 0
    rvd_value = 0
    hd95_value = 0
    recall_value = 0
    miou_value = 0
    sen_value = 0
    f1_value = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                
                # compute other metrics
                iou_value += iou_score(mask_pred, mask_true)
                prec_value += precision_score(mask_pred, mask_true)
                acc_value += ((mask_pred == mask_true).sum() / torch.numel(mask_true)).item()
                voe_value += voe_score(mask_pred.bool(), mask_true.bool())
                rvd_value += rvd_score(mask_pred, mask_true)
                hd95_value += hausdorff_95(mask_pred.squeeze(), mask_true.squeeze())
                # compute recall
                actual_positives = mask_true.sum()
                true_positives = (mask_pred * mask_true).sum()
                recall_value += (true_positives + 1e-6) / (actual_positives + 1e-6)
                
                # compute additional metrics
                miou_value += iou_value  # 对于二分类任务，mIoU等于IoU
                sen_value += recall_value  # 敏感度等于召回率
                f1_value += 2 * (prec_value * recall_value) / (prec_value + recall_value + 1e-6)  # F1分数
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                
                # compute other metrics (for multi-class, we focus on class 1)
                iou_value += iou_score(mask_pred[:, 1], mask_true[:, 1])
                prec_value += precision_score(mask_pred[:, 1], mask_true[:, 1])
                acc_value += ((mask_pred[:, 1] == mask_true[:, 1]).sum() / torch.numel(mask_true[:, 1])).item()
                voe_value += voe_score(mask_pred[:, 1].bool(), mask_true[:, 1].bool())
                rvd_value += rvd_score(mask_pred[:, 1], mask_true[:, 1])
                hd95_value += hausdorff_95(mask_pred[:, 1].squeeze(), mask_true[:, 1].squeeze())
                # compute recall
                actual_positives = mask_true[:, 1].sum()
                true_positives = (mask_pred[:, 1] * mask_true[:, 1]).sum()
                recall_value += (true_positives + 1e-6) / (actual_positives + 1e-6)
                
                # compute additional metrics
                miou_value += iou_value  # 对于多分类任务，mIoU等于IoU
                sen_value += recall_value  # 敏感度等于召回率
                f1_value += 2 * (prec_value * recall_value) / (prec_value + recall_value + 1e-6)  # F1分数

    net.train()
    
    # 计算平均值
    dice_score = dice_score / max(num_val_batches, 1)
    iou_value = iou_value / max(num_val_batches, 1)
    prec_value = prec_value / max(num_val_batches, 1)
    acc_value = acc_value / max(num_val_batches, 1)
    voe_value = voe_value / max(num_val_batches, 1)
    rvd_value = rvd_value / max(num_val_batches, 1)
    hd95_value = hd95_value / max(num_val_batches, 1)
    recall_value = recall_value / max(num_val_batches, 1)
    miou_value = miou_value / max(num_val_batches, 1)
    sen_value = sen_value / max(num_val_batches, 1)
    f1_value = f1_value / max(num_val_batches, 1)

    return dice_score, iou_value, prec_value, acc_value, voe_value, rvd_value, hd95_value, recall_value, miou_value, sen_value, f1_value
