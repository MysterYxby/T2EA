import os
from PIL import Image

input_folder = "D:/Result/detection/U2Fusion"  # 输入文件夹路径
output_folder = "Output/demo"  # 输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".bmp") or file.endswith(".tiff"):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".png")
            # 打开图像文件
            img = Image.open(input_path)
            # 转换并保存为PNG格式
            img.save(output_path, "PNG")