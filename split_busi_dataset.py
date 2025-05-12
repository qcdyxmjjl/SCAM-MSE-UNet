import os
import shutil
import random
from pathlib import Path
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_directory_structure(base_dir):
    """创建数据集目录结构"""
    splits = ['train', 'val', 'test']
    subdirs = ['images', 'masks']
    
    for split in splits:
        for subdir in subdirs:
            path = base_dir / split / subdir
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f'创建目录: {path}')

def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    将数据集按比例分割为训练集、验证集和测试集
    
    Args:
        input_dir (Path): 输入目录路径
        output_dir (Path): 输出目录路径
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
    """
    # 获取所有图像文件
    image_files = list((input_dir / 'images').glob('*.*'))
    mask_files = list((input_dir / 'masks').glob('*.*'))
    
    # 确保图像和掩码文件数量匹配
    if len(image_files) != len(mask_files):
        raise ValueError(f'图像和掩码文件数量不匹配: {len(image_files)} vs {len(mask_files)}')
    
    # 随机打乱文件列表
    combined = list(zip(image_files, mask_files))
    random.shuffle(combined)
    
    # 计算分割点
    total = len(combined)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 分割数据集
    train_set = combined[:train_end]
    val_set = combined[train_end:val_end]
    test_set = combined[val_end:]
    
    # 复制文件到相应目录
    for split, dataset in [('train', train_set), ('val', val_set), ('test', test_set)]:
        for img_file, mask_file in dataset:
            # 复制图像
            shutil.copy2(img_file, output_dir / split / 'images' / img_file.name)
            # 复制掩码
            shutil.copy2(mask_file, output_dir / split / 'masks' / mask_file.name)
        
        logging.info(f'{split}集: {len(dataset)} 对图像')

def main():
    """主函数"""
    setup_logging()
    
    # 设置目录
    input_dir = Path('data/BUSI/train')
    output_dir = Path('data/BUSI/split')
    
    try:
        # 创建输出目录结构
        create_directory_structure(output_dir)
        
        # 分割数据集
        split_dataset(input_dir, output_dir)
        
        logging.info('数据集分割完成！')
        logging.info(f'训练集、验证集和测试集已保存到: {output_dir}')
        
    except Exception as e:
        logging.error(f'处理过程中出现错误: {str(e)}')

if __name__ == '__main__':
    main() 