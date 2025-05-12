import matplotlib.pyplot as plt
from pathlib import Path

class MetricsLogger:
    def __init__(self):
        self.epochs = []
        # 验证集指标
        self.val_dice = []
        self.val_iou = []
        self.val_prec = []
        self.val_recall = []
        self.val_acc = []
        self.val_voe = []
        self.val_rvd = []
        self.val_hd95 = []
        # 测试集指标
        self.test_dice = []
        self.test_iou = []
        self.test_prec = []
        self.test_recall = []
        self.test_acc = []
        self.test_voe = []
        self.test_rvd = []
        self.test_hd95 = []
        # 训练损失
        self.train_loss = []
        
    def update(self, epoch, metrics_dict):
        """更新所有指标的值"""
        self.epochs.append(epoch)
        # 更新验证集指标
        self.val_dice.append(metrics_dict['val Dice'])
        self.val_iou.append(metrics_dict['val IoU'])
        self.val_prec.append(metrics_dict['val Precision'])
        self.val_recall.append(metrics_dict['val Recall'])
        self.val_acc.append(metrics_dict['val Accuracy'])
        self.val_voe.append(metrics_dict['val VOE'])
        self.val_rvd.append(metrics_dict['val RVD'])
        self.val_hd95.append(metrics_dict['val HD95'])
        # 更新测试集指标
        self.test_dice.append(metrics_dict['test Dice'])
        self.test_iou.append(metrics_dict['test IoU'])
        self.test_prec.append(metrics_dict['test Precision'])
        self.test_recall.append(metrics_dict['test Recall'])
        self.test_acc.append(metrics_dict['test Accuracy'])
        self.test_voe.append(metrics_dict['test VOE'])
        self.test_rvd.append(metrics_dict['test RVD'])
        self.test_hd95.append(metrics_dict['test HD95'])
        # 更新训练损失
        self.train_loss.append(metrics_dict['train loss'])
    
    def plot_metrics(self, save_dir, model_name):
        """绘制所有指标的图表"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制主要指标（Dice, IoU）
        plt.figure(figsize=(12, 6))
        plt.plot(self.epochs, self.val_dice, 'b-', label='验证集 Dice')
        plt.plot(self.epochs, self.test_dice, 'b--', label='测试集 Dice')
        plt.plot(self.epochs, self.val_iou, 'r-', label='验证集 IoU')
        plt.plot(self.epochs, self.test_iou, 'r--', label='测试集 IoU')
        plt.title(f'{model_name} - Dice和IoU指标')
        plt.xlabel('轮次')
        plt.ylabel('分数')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_dir / 'dice_iou_metrics.png')
        plt.close()
        
        # 绘制准确率相关指标
        plt.figure(figsize=(12, 6))
        plt.plot(self.epochs, self.val_prec, 'g-', label='验证集 Precision')
        plt.plot(self.epochs, self.test_prec, 'g--', label='测试集 Precision')
        plt.plot(self.epochs, self.val_recall, 'm-', label='验证集 Recall')
        plt.plot(self.epochs, self.test_recall, 'm--', label='测试集 Recall')
        plt.plot(self.epochs, self.val_acc, 'y-', label='验证集 Accuracy')
        plt.plot(self.epochs, self.test_acc, 'y--', label='测试集 Accuracy')
        plt.title(f'{model_name} - 准确率相关指标')
        plt.xlabel('轮次')
        plt.ylabel('分数')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_dir / 'accuracy_metrics.png')
        plt.close()
        
        # 绘制其他评估指标
        plt.figure(figsize=(12, 6))
        plt.plot(self.epochs, self.val_voe, 'c-', label='验证集 VOE')
        plt.plot(self.epochs, self.test_voe, 'c--', label='测试集 VOE')
        plt.plot(self.epochs, self.val_rvd, 'k-', label='验证集 RVD')
        plt.plot(self.epochs, self.test_rvd, 'k--', label='测试集 RVD')
        plt.plot(self.epochs, self.val_hd95, 'r-', label='验证集 HD95')
        plt.plot(self.epochs, self.test_hd95, 'r--', label='测试集 HD95')
        plt.title(f'{model_name} - 其他评估指标')
        plt.xlabel('轮次')
        plt.ylabel('分数')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_dir / 'other_metrics.png')
        plt.close()
        
        # 绘制训练损失
        plt.figure(figsize=(12, 6))
        plt.plot(self.epochs, self.train_loss, 'b-', label='训练损失')
        plt.title(f'{model_name} - 训练损失')
        plt.xlabel('轮次')
        plt.ylabel('损失值')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_dir / 'training_loss.png')
        plt.close() 