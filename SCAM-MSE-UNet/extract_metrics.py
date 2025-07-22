import pandas as pd
import re
from pathlib import Path

def extract_metrics(log_file='output.log', output_file='dice_iou_metrics.csv'):
    print(f"正在从 {log_file} 提取 Dice Score 和 IoU Score 数据...")
    
    all_metrics = []
    current_metrics = {}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 检测新的epoch
                epoch_match = re.search(r'Epoch (\d+)/', line)
                if epoch_match:
                    if current_metrics:
                        all_metrics.append(current_metrics)
                    current_metrics = {'epoch': int(epoch_match.group(1))}
                
                # 检测验证/测试阶段
                if 'Validation Metrics:' in line:
                    current_metrics['is_validation'] = True
                elif 'Test Metrics:' in line:
                    current_metrics['is_validation'] = False
                
                # 只提取 Dice Score 和 IoU Score
                metric_match = re.search(r'INFO: (Dice Score|IoU Score):\s+([-+]?\d*\.\d+|\d+)', line)
                if metric_match and 'is_validation' in current_metrics:
                    metric_name = metric_match.group(1).strip()
                    metric_value = float(metric_match.group(2))
                    
                    prefix = 'val_' if current_metrics['is_validation'] else 'test_'
                    current_metrics[f'{prefix}{metric_name}'] = metric_value
        
        if current_metrics:
            all_metrics.append(current_metrics)
        
        # 将数据保存为CSV文件
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            if 'is_validation' in df.columns:
                df = df.drop('is_validation', axis=1)
            
            # 只保留需要的列
            columns_to_keep = ['epoch']
            for prefix in ['val_', 'test_']:
                for metric in ['Dice Score', 'IoU Score']:
                    col = f'{prefix}{metric}'
                    if col in df.columns:
                        columns_to_keep.append(col)
            
            df = df[columns_to_keep]
            df.to_csv(output_file, index=False)
            print(f"Dice Score 和 IoU Score 数据已成功保存到 {output_file}")
            
            # 打印统计信息
            print("\n统计信息:")
            for column in df.columns:
                if column != 'epoch':
                    print(f"{column}:")
                    print(f"  平均值: {df[column].mean():.4f}")
                    print(f"  最大值: {df[column].max():.4f}")
                    print(f"  最小值: {df[column].min():.4f}")
                    print()
        else:
            print("未找到任何指标数据")
            
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")

if __name__ == '__main__':
    extract_metrics() 