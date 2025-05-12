import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import wandb
from unet.cenet import CENet
from unet.basesaf import basesaf
from unet.ldcunet import ldcunet
from unet.LDCUNet import LDCUNet
from evaluate_plus import evaluate_plus
from unet.our_model import my_model
from unet.CSANUNet import CSANUNet
from unet.CSAUNet import CSAUNet
from unet import UNet
from unet.attention_unet import AttU_Net
from unet.r2unet import R2U_Net
from unet.segnet import SegNet
from unet.unetplusplus import UNetPlusPlus
from unet.unetpp import NestedUNet
from unet.amunet import AMUNet
from utils import metrics_logger
from utils.metrics_logger import MetricsLogger
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from plot import loss_plot, metrics_plot

dir_img = Path('./data/cell/train/images/')
dir_mask = Path('./data/cell/train/mask/')
dir_val_img = Path('./data/cell/val/images/')
dir_val_mask = Path('./data/cell/val/mask/')
dir_test_img = Path('./data/cell/test/images/')
dir_test_mask = Path('./data/cell/test/mask/')
dir_checkpoint = Path('check_points_all/checkpoints_unetpp_celltest')

def train_model(
        model,
        device,
        epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = True,
        weight_decay: float = 1e-5,
        momentum: float = 0.9,
        gradient_clipping: float = 1.0,
):
    # 1. 检查是否存在验证集和测试集
    has_val_set = dir_val_img.exists() and dir_val_mask.exists() and len(list(dir_val_img.glob('*'))) > 0
    has_test_set = dir_test_img.exists() and dir_test_mask.exists() and len(list(dir_test_img.glob('*'))) > 0

    # 2. 创建数据集
    try:
        # 首先尝试使用CarvanaDataset
        original_dataset = CarvanaDataset(dir_img, dir_mask, img_scale, is_train=True)
    except (AssertionError, RuntimeError, IndexError):
        try:
            # 如果失败，尝试使用BasicDataset，先不带后缀
            original_dataset = BasicDataset(dir_img, dir_mask, img_scale, is_train=True)
        except (AssertionError, RuntimeError, IndexError):
            # 如果还是失败，尝试使用BasicDataset，带_mask后缀
            original_dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='_mask', is_train=True)

    if has_val_set and has_test_set:
        # 如果有独立的验证集和测试集，直接使用
        try:
            val_dataset = CarvanaDataset(dir_val_img, dir_val_mask, img_scale, is_train=False)
            test_dataset = CarvanaDataset(dir_test_img, dir_test_mask, img_scale, is_train=False)
        except (AssertionError, RuntimeError, IndexError):
            try:
                val_dataset = BasicDataset(dir_val_img, dir_val_mask, img_scale, is_train=False)
                test_dataset = BasicDataset(dir_test_img, dir_test_mask, img_scale, is_train=False)
            except (AssertionError, RuntimeError, IndexError):
                val_dataset = BasicDataset(dir_val_img, dir_val_mask, img_scale, mask_suffix='_mask', is_train=False)
                test_dataset = BasicDataset(dir_test_img, dir_test_mask, img_scale, mask_suffix='_mask', is_train=False)
        
        train_dataset = original_dataset
    else:
        # 如果没有独立的验证集或测试集，则从原始数据集中划分
        # 获取所有图像文件名
        all_indices = list(range(len(original_dataset)))
        random.seed(42)  # 设置随机种子确保可重复性
        random.shuffle(all_indices)
        
        # 按8:1:1的比例划分
        n_total = len(original_dataset)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = n_total - n_train - n_val
        
        # 创建不重叠的数据集
        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:n_train + n_val]
        test_indices = all_indices[n_train + n_val:]
        
        from torch.utils.data import Subset
        train_dataset = Subset(original_dataset, train_indices)
        val_dataset = Subset(original_dataset, val_indices)
        test_dataset = Subset(original_dataset, test_indices)

    # 3. 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp),
        allow_val_change=True  # 允许更新配置值
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Test size:       {len(test_dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Using val set:   {has_val_set}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(model.parameters(),
                         lr=learning_rate,
                         weight_decay=weight_decay,
                         foreach=True)
    
    # 修改学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                    patience=10,  # 增加耐心值
                                                    factor=0.5,   # 每次降低一半
                                                    min_lr=1e-6,  # 最小学习率
                                                    verbose=True)

    # 根据类别数选择合适的损失函数
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    global_step = 0
    best_dice_score = 0.0

    # 初始化记录列表
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)  # CrossEntropyLoss会处理维度问题
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # 确保这部分代码的缩进与for epoch对齐
        print(f"\nEpoch {epoch} finished. Starting evaluation...")
        logging.info(f"\nEpoch {epoch} finished. Starting evaluation...")

        # 在评估之前
        model.eval()  # 切换到评估模式
        with torch.no_grad():  # 不计算梯度
            logging.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            
            logging.info(f"Starting evaluation for epoch {epoch}...")
            try:
                # 使用evaluate_plus.py进行验证集评估
                val_dice, val_iou, val_prec, val_acc, val_voe, val_rvd, val_hd95, val_recall, val_miou, val_sen, val_f1 = evaluate_plus(model, val_loader, device, amp)
                
                # 使用evaluate_plus.py进行测试集评估
                test_dice, test_iou, test_prec, test_acc, test_voe, test_rvd, test_hd95, test_recall, test_miou, test_sen, test_f1 = evaluate_plus(model, test_loader, device, amp)
                
                logging.info("Evaluation completed")
            except Exception as e:
                logging.error(f"Error during evaluation: {str(e)}")
                raise e
        model.train()  # 评估后切换回训练模式
        
        # 显示评估指标
        model_name = model.__class__.__name__
        logging.info('=' * 50)
        logging.info(f'Model: {model_name}')
        logging.info('=' * 50)
        
        logging.info('-' * 50)
        logging.info(f'Epoch {epoch} - Validation Metrics:')
        logging.info('-' * 50)
        logging.info(f'Dice Score:     {val_dice:.4f}')
        logging.info(f'IoU Score:      {val_iou:.4f}')
        logging.info(f'Precision:      {val_prec:.4f}')
        logging.info(f'Recall:         {val_recall:.4f}')
        logging.info(f'Accuracy:       {val_acc:.4f}')
        logging.info(f'VOE:            {val_voe:.4f}')
        logging.info(f'RVD:            {val_rvd:.4f}')
        logging.info(f'HD95:           {val_hd95:.4f}')
        logging.info(f'mIoU:           {val_miou:.4f}')
        logging.info(f'Sensitivity:    {val_sen:.4f}')
        logging.info(f'F1 Score:       {val_f1:.4f}')
        logging.info('-' * 50)

        logging.info('-' * 50)
        logging.info(f'Epoch {epoch} - Test Metrics:')
        logging.info('-' * 50)
        logging.info(f'Dice Score:     {test_dice:.4f}')
        logging.info(f'IoU Score:      {test_iou:.4f}')
        logging.info(f'Precision:      {test_prec:.4f}')
        logging.info(f'Recall:         {test_recall:.4f}')
        logging.info(f'Accuracy:       {test_acc:.4f}')
        logging.info(f'VOE:            {test_voe:.4f}')
        logging.info(f'RVD:            {test_rvd:.4f}')
        logging.info(f'HD95:           {test_hd95:.4f}')
        logging.info(f'mIoU:           {test_miou:.4f}')
        logging.info(f'Sensitivity:    {test_sen:.4f}')
        logging.info(f'F1 Score:       {test_f1:.4f}')
        logging.info('-' * 50)

        scheduler.step(val_dice)  # 使用验证集的dice分数来调整学习率

        # 记录本轮的loss和指标
        avg_loss = epoch_loss/len(train_loader)
        loss_list.append(avg_loss)
        
        # 根据类型进行相应的处理
        def process_metric(metric):
            if torch.is_tensor(metric):
                return metric.cpu().detach()
            return metric
            
        iou_list.append(process_metric(test_iou))
        dice_list.append(process_metric(test_dice))
        hd_list.append(process_metric(test_hd95))

        # 创建保存图像的目录
        save_dir = os.path.join('result', model.__class__.__name__, str(batch_size), str(epochs))
        os.makedirs(save_dir, exist_ok=True)

        # 绘制并保存loss曲线
        loss_plot(args, loss_list)
        plt.savefig(os.path.join(save_dir, 'loss.png'))
        plt.close()

        # 绘制并保存iou和dice曲线
        metrics_plot(args, 'iou&dice', iou_list, dice_list)
        plt.savefig(os.path.join(save_dir, 'iou_dice.png'))
        plt.close()

        # 绘制并保存hausdorff distance曲线
        metrics_plot(args, 'hd', hd_list)
        plt.savefig(os.path.join(save_dir, 'hd.png'))
        plt.close()

        # 记录到wandb
        metrics_dict = {
            'Model': model_name,
            'learning rate': optimizer.param_groups[0]['lr'],
            'train loss': avg_loss,
            # 验证集指标
            'val Dice': val_dice,
            'val IoU': val_iou,
            'val Precision': val_prec,
            'val Recall': val_recall,
            'val Accuracy': val_acc,
            'val VOE': val_voe,
            'val RVD': val_rvd,
            'val HD95': val_hd95,
            'val mIoU': val_miou,
            'val Sensitivity': val_sen,
            'val F1 Score': val_f1,
            # 测试集指标
            'test Dice': test_dice,
            'test IoU': test_iou,
            'test Precision': test_prec,
            'test Recall': test_recall,
            'test Accuracy': test_acc,
            'test VOE': test_voe,
            'test RVD': test_rvd,
            'test HD95': test_hd95,
            'test mIoU': test_miou,
            'test Sensitivity': test_sen,
            'test F1 Score': test_f1,
            'best validation Dice': best_dice_score,
        }
        
        # 添加图像样本
        metrics_dict.update({
            'images': wandb.Image(images[0].cpu()),
            'masks': {
                'true': wandb.Image(true_masks[0].float().cpu()),
                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
            },
            'step': global_step,
            'epoch': epoch,
        })
        
        experiment.log(metrics_dict)

        # 保存最优权重
        if val_dice > best_dice_score:  # 使用验证集的dice分数来保存最佳模型
            best_dice_score = val_dice
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = original_dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'best_model.pth'))
            print(f'New best model saved! Best Dice score: {best_dice_score:.4f}')
            logging.info(f'New best model saved! Best Dice score: {best_dice_score:.4f}')

        # 每25轮保存一次权重
        if epoch % 25 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = original_dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            print(f'Checkpoint {epoch} saved!')
            logging.info(f'Checkpoint {epoch} saved!')

        sys.stdout.flush()

    # 在训练结束后绘制图表
    metrics_logger_instance = MetricsLogger()  # 创建MetricsLogger实例
    metrics_logger_instance.plot_metrics(dir_checkpoint, model_name)

    # 训练完成后，加载最优权重进行测试集预测
    logging.info('训练完成，开始进行测试集预测...')
    
    # 创建保存预测结果的目录
    save_dir = os.path.join('predictions', model.__class__.__name__)
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载最优权重
    best_model_path = str(dir_checkpoint / 'best_model.pth')
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'加载最优模型权重: {best_model_path}')
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, 
                           batch_size=1,  # 每次处理一张图片
                           shuffle=False,
                           num_workers=0,
                           pin_memory=True)
    
    # 对测试集进行预测
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='生成预测结果')):
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            
            # 预测
            masks_pred = model(images)
            
            # 处理预测结果
            if model.n_classes == 1:
                probs = torch.sigmoid(masks_pred)
                masks_pred = (probs > 0.5).float()
            else:
                masks_pred = F.softmax(masks_pred, dim=1)
                masks_pred = masks_pred.argmax(dim=1)
            
            # 将预测结果转换为图像
            pred_mask = masks_pred[0].cpu().numpy()
            true_mask = true_masks[0].cpu().numpy()
            original_img = images[0].cpu().numpy().transpose(1, 2, 0)
            
            # 创建图像网格
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 显示原始图像
            axes[0].imshow(original_img)
            axes[0].set_title('原始图像')
            axes[0].axis('off')
            
            # 显示真实掩码
            axes[1].imshow(true_mask, cmap='gray')
            axes[1].set_title('真实掩码')
            axes[1].axis('off')
            
            # 显示预测掩码
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title('预测掩码')
            axes[2].axis('off')
            
            # 保存图像
            plt.savefig(os.path.join(save_dir, f'prediction_{i+1}.png'))
            plt.close()
            
            # 每10张图像记录一次日志
            if (i + 1) % 10 == 0:
                logging.info(f'已处理 {i+1} 张测试图像')
    
    logging.info(f'预测完成！结果保存在: {save_dir}')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=5e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--arch', type=str, default='CSAUNet', help='Model architecture')
    parser.add_argument('--dataset', type=str, default='liver', help='Dataset name')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    
    model = CENet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
        #initial_channels=initial_channels  # 添加initial_channels参数
    )
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=True  # 直接启用混合精度训练
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('检测到显存不足！尝试以下解决方案：')
        logging.error('1. 减小batch_size')
        logging.error('2. 减小图像尺寸 (使用--scale参数)')
        logging.error('3. 已启用混合精度训练')
        
        # 自动调整batch_size和scale
        new_batch_size = args.batch_size // 2
        new_scale = args.scale * 0.8
        
        logging.info(f'自动调整参数: batch_size={new_batch_size}, scale={new_scale:.2f}')
        
        torch.cuda.empty_cache()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=new_batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=new_scale,
            amp=True  # 保持混合精度训练启用
        )
