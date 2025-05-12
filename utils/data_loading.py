import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_training_augmentation():
    """获取训练时的数据增强（简化版）"""
    train_transform = [
        A.RandomRotate90(p=0.5),        # 随机90度旋转
        A.HorizontalFlip(p=0.5),      # 水平翻转
        A.VerticalFlip(p=0.5),        # 垂直翻转
        A.ShiftScaleRotate(             # 随机平移、缩放、旋转
            shift_limit=0.0625, 
            scale_limit=0.1, 
            rotate_limit=45, 
            p=0.5
        ),
        A.OneOf([                       # 随机选择一种像素级变换
            A.RandomBrightnessContrast(p=0.5),  # 亮度和对比度调整
            A.RandomGamma(p=0.5),               # 伽马校正
        ], p=0.3),
        A.Resize(height=512, width=512),  # 确保最终大小为512x512
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    """获取验证时的确定性增强"""
    return A.Compose([
        A.Resize(height=512, width=512),  # 统一尺寸为512x512
    ])

def get_test_time_augmentation():
    """获取测试时增强（简化版TTA）"""
    return A.Compose([
        A.HorizontalFlip(p=1.0),        # 水平翻转
        A.VerticalFlip(p=1.0),          # 垂直翻转
        A.Resize(height=512, width=512),  # 确保最终大小为512x512
    ])

def get_preprocessing():
    """获取预处理转换"""
    _transform = [
        A.Resize(height=512, width=512),  # 统一调整图像大小为512x512
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(_transform)

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '_centerline_overlay', is_train: bool = False, use_tta: bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.is_train = is_train
        self.use_tta = use_tta

        # 检查目录是否存在
        if not self.images_dir.exists():
            raise RuntimeError(f'图像目录不存在: {self.images_dir}')
        if not self.masks_dir.exists():
            raise RuntimeError(f'掩码目录不存在: {self.masks_dir}')

        # 获取所有图像文件
        self.ids = [file.stem for file in self.images_dir.glob('*') if not file.name.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'在目录中没有找到图像: {self.images_dir}')

        # 检查每个图像是否都有对应的掩码，同时检查是否需要使用后缀
        valid_ids = []
        use_suffix = False
        
        # 首先检查是否存在带后缀的掩码文件
        for img_id in self.ids:
            mask_files_with_suffix = list(self.masks_dir.glob(f'{img_id}{self.mask_suffix}.*'))
            if len(mask_files_with_suffix) == 1:
                use_suffix = True
                break

        # 根据是否使用后缀来验证文件
        for img_id in self.ids:
            img_files = list(self.images_dir.glob(f'{img_id}.*'))
            if use_suffix:
                mask_files = list(self.masks_dir.glob(f'{img_id}{self.mask_suffix}.*'))
            else:
                mask_files = list(self.masks_dir.glob(f'{img_id}.*'))
            
            if len(img_files) == 1 and len(mask_files) == 1:
                valid_ids.append(img_id)
            else:
                logging.warning(f'跳过 {img_id}: 图像或掩码文件不完整')
        
        self.ids = valid_ids
        self.use_suffix = use_suffix
        
        if not self.ids:
            raise RuntimeError('没有找到有效的图像-掩码对')

        logging.info(f'创建数据集，共有 {len(self.ids)} 对图像和掩码')
        logging.info(f'掩码文件{"使用" if use_suffix else "不使用"}后缀: {self.mask_suffix}')
        logging.info(f'使用 {"训练" if is_train else "验证"} 模式的数据增强')

        # 设置增强方法
        if is_train:
            self.augmentation = get_training_augmentation()
        elif use_tta:
            self.augmentation = get_test_time_augmentation()
        else:
            self.augmentation = get_validation_augmentation()

        self.preprocessing = get_preprocessing()

        # 扫描掩码文件确定唯一值
        logging.info('扫描掩码文件以确定唯一值...')
        unique_values = set()
        for img_id in tqdm(self.ids):
            if self.use_suffix:
                mask_file = list(self.masks_dir.glob(f'{img_id}{self.mask_suffix}.*'))[0]
            else:
                mask_file = list(self.masks_dir.glob(f'{img_id}.*'))[0]
            mask = np.asarray(Image.open(mask_file))
            unique_values.update(np.unique(mask).tolist())
        
        self.mask_values = sorted(list(unique_values))
        logging.info(f'掩码中的唯一值: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        
        img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[img[:, :, 0] == v] = i
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            
            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        if self.use_suffix:
            mask_file = list(self.masks_dir.glob(f'{name}{self.mask_suffix}.*'))[0]
        else:
            mask_file = list(self.masks_dir.glob(f'{name}.*'))[0]
        img_file = list(self.images_dir.glob(f'{name}.*'))[0]
        
        # 加载图像和掩码
        img = Image.open(img_file)
        mask = Image.open(mask_file)

        # 转换为numpy数组以用于albumentations
        img_np = np.array(img)
        mask_np = np.array(mask)
        
        # 如果是灰度图，转换为3通道
        if len(img_np.shape) == 2:
            img_np = np.stack((img_np,)*3, axis=-1)
        elif img_np.shape[2] == 1:
            img_np = np.concatenate((img_np,)*3, axis=-1)
        
        # 确保掩码值为0和1
        if mask_np.max() > 1:
            mask_np = (mask_np > 127).astype(np.uint8)  # 使用127作为阈值
        
        # 打印调试信息
        if idx == 0:  # 只打印第一个样本的信息
            print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}, range: [{img_np.min()}, {img_np.max()}]")
            print(f"Mask shape: {mask_np.shape}, dtype: {mask_np.dtype}, unique values: {np.unique(mask_np)}")

        # 应用数据增强
        if self.augmentation:
            augmented = self.augmentation(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']

        # 应用预处理
        preprocessed = self.preprocessing(image=img_np, mask=mask_np)
        img_tensor = preprocessed['image']
        mask_tensor = preprocessed['mask'].long()

        return {
            'image': img_tensor,
            'mask': mask_tensor
        }

    @staticmethod
    @lru_cache(maxsize=None)
    def load_image(filename):
        ext = filename.suffix
        if ext in ['.npz', '.npy']:
            mask = np.load(filename)
            return mask
        elif ext in ['.pt', '.pth']:
            mask = torch.load(filename)
            return mask
        else:
            return Image.open(filename)


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1, is_train=False, use_tta=False):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_centerline_overlay', 
                        is_train=is_train, use_tta=use_tta)


