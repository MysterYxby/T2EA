import torch
from TEM import  Taylor_Encoder
from PIL import Image
from utils import tensor_rgb2ycbcr,tensor_ycbcr2rgb,save_PIL,transform_img,save_seg
from ssim_loss import ssim
import os

def Taylor_Metirc(root, model_root):
    Net = Taylor_Encoder()
    Net.load_state_dict(torch.load(model_root, map_location=torch.device('cpu')))
    Net.eval()  # 设置为评估模式

    # 测试数据路径
    root = os.path.join(root)
    SSIM_loss = []

    # 读取文件名
    filenames_ir = os.listdir(root)
    for i in range(len(filenames_ir)):
        filepath = os.path.join(root, filenames_ir[i])
        img = Image.open(filepath).convert('L')
        img = transform_img(img)
        
        with torch.no_grad():  # 使用no_grad()包裹前向传播
            result, _ = Net(img, 2)
        
        SSIM_loss.append(ssim(result, img))
        
        del result, img  # 删除不再需要的变量

    tensor_data = torch.tensor(SSIM_loss)

    # 计算平均值
    mean_value = torch.mean(tensor_data)

    print("平均值为:", mean_value.item())
    return 0


if __name__ == '__main__':
    Taylor_Metirc(
        root = 'datasets/MSRS/Visible/test/MSRS',
        model_root = 'code/model/Taylor/RV5/model100.pt'
    )
    