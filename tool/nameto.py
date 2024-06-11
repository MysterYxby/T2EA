# import os

# def rename_images_in_folder(folder_path):
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
#             # 构建原始文件路径和新文件路径
#             old_file_path = os.path.join(folder_path, file_name)
#             new_file_name = file_name[1:]  # 删除第一个字符
#             new_file_path = os.path.join(folder_path, new_file_name)

#             # 重命名文件
#             os.rename(old_file_path, new_file_path)
#             print(f"重命名 {old_file_path} 为 {new_file_path}")


# if __name__ =='__main__':
#     names = ['T2EA','label','U2Fusion','DRF','DenseFuse','MSPIF','FusionGAN','Infrared','Visible','GANMcC','UMF-CMGR']   
#     # names = ['DRF','DenseFuse','MSPIF','FusionGAN','Infrared','Visible','GANMcC','UMF-CMGR'] 
#     for name in names:

#         folder_path = "D:/Result/Segmentation/" + name
#         rename_images_in_folder(folder_path)

import os
from PIL import Image

# 指定文件夹路径
folder_path = "D:/Result/Segmentation/U2Fusion"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否为图片文件
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 打开图像文件
        image = Image.open(file_path)
        
        # 生成 PNG 文件的文件名
        png_filename = os.path.splitext(filename)[0] + ".png"
        png_file_path = os.path.join(folder_path, png_filename)
        
        # 将图像保存为 PNG 格式
        image.save(png_file_path, "PNG")
        
        print(f"已将 {filename} 转换为 {png_filename}")

print("所有图片已成功转换为 PNG 格式。")