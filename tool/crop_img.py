import os
from PIL import Image
from torchvision import transforms

# 设置图片的文件夹路径和保存剪裁图片的路径
input_folder = 'datasets/MSRS/Visible/test/MSRS'
output_folder = 'datasets/MSRS/Gray/Visible/'

# 创建保存剪裁后图片的文件夹（如果不存在的话）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置随机裁剪的大小
crop_size = (480, 640)

# 创建一个随机裁剪的transform
random_crop = transforms.RandomCrop(crop_size)

# # 遍历文件夹中的所有图片文件
# for file_name in os.listdir(input_folder):
#     if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
#         # 加载图片
#         image_path = os.path.join(input_folder, file_name)
#         image = Image.open(image_path)
        
#         # 应用随机裁剪
#         cropped_image = random_crop(image)
        
#         # 保存裁剪后的图片
#         output_path = os.path.join(output_folder, 'v_' + file_name)
#         cropped_image.save(output_path)


# 遍历文件夹中的所有图片文件
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # 加载图片
        image_path = os.path.join(input_folder, file_name)
        image = Image.open(image_path).convert('L')
        
        # 应用随机裁剪
        # cropped_image = random_crop(image)
        
        # 保存裁剪后的图片
        output_path = os.path.join(output_folder, file_name)
        image.save(output_path)
print("All images have been randomly cropped and saved to", output_folder)