import os
import shutil

# 设置路径
source_dir = r"D:\science\DataSet\ISIC-2017\test\ISIC-2017_Validation_Data"         # 当前目录（可以修改为你的源目录）
images_dir = r"D:\science\DataSet\ISIC-2017\test\images"          # 普通图片存放目录
superpixels_dir = r"D:\science\DataSet\ISIC-2017\test\superpixels"  # 超像素文件存放目录

# 创建目标文件夹
os.makedirs(images_dir, exist_ok=True)
os.makedirs(superpixels_dir, exist_ok=True)

# 遍历当前目录下的所有文件
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    
    # 跳过目录，只处理文件
    if os.path.isfile(file_path):
        # 检查文件名是否包含"_superpixels"
        if "_superpixels" in filename:
            # 移动超像素文件
            dest = os.path.join(superpixels_dir, filename)
            shutil.move(file_path, dest)
            print(f"Moved superpixels file: {filename}")
        else:
            # 移动普通图片文件
            dest = os.path.join(images_dir, filename)
            shutil.move(file_path, dest)
            print(f"Moved image file: {filename}")

print("File organization completed!")