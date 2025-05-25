import torch
from network.TEM import Taylor_Encoder
import os
from network.FusionNet import FusionModel
from PIL import Image
from utils import transform_img,tensor_rgb2ycbcr,save_PIL,tensor_ycbcr2rgb

def fusion_test(vis,ir,save_path,Net,Fusion,device,filenames_vis,i):
    _,c,_,_ = vis.size()
    with torch.no_grad():
        if c == 3:
            #融合图像应该为彩色
            gray, img1_cb, img1_cr = tensor_rgb2ycbcr(vis)
                #Taylor Encoder
            _,y_vis = Net(gray.to(device),2)
            _,y_ir  = Net(ir.to(device),2)
            result = Fusion(y_vis,y_ir)

            #颜色恢复
            result =  tensor_ycbcr2rgb(result, img1_cb.to(device), img1_cr.to(device))
            # 保存融合图像
            save_PIL(result,os.path.join(save_path,filenames_vis[i]))
            print('the {}th pair of image is  Ok!'.format(str(i+1)))
        else:
            #Taylor Encoder
            _,y_vis = Net(vis.to(device),2)
            _,y_ir  = Net(ir.to(device),2)
            result = Fusion(y_vis,y_ir)
        save_PIL(result,os.path.join(save_path,filenames_vis[i]))  
        print('the {}th pair of image is  Ok!'.format(str(i+1)))
        
    return 0

def test_Fusion(ir_root,vis_root,save_path):
    model_fusion = 'model/Fusion.pt'
    model_Taylor = 'model/Taylor.pt'
     #判断保存图像路径是否存在
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #模型加载
    Net = Taylor_Encoder()
    Net.load_state_dict(torch.load(model_Taylor, map_location=device))
    Net = Net.to(device)
    Net = Net.eval()
    print('Taylor Decomposition is successfully loaded!')
    Fusion = FusionModel()
    Fusion.load_state_dict(torch.load(model_fusion, map_location=device))
    Fusion = Fusion.to(device)
    Fusion = Fusion.eval()
    print('FusionNet is successfully loaded!')
    
    #读取文件
    filenames_ir = os.listdir(ir_root)
    filenames_vis = os.listdir(vis_root)
    print('The number of total images is {} pairs'.format(str(len(filenames_ir))))
    for i in range (len(filenames_ir)):
        filepath_ir = os.path.join(ir_root,filenames_ir[i])
        ir = Image.open(filepath_ir).convert('L')
        ir = transform_img(ir)
        filepath_vis = os.path.join(vis_root,filenames_vis[i])
        vis = Image.open(filepath_vis)
        vis = transform_img(vis)
        fusion_test(vis,ir,save_path,Net,Fusion,device,filenames_vis,i) 

if __name__ =='__main__':
    test_Fusion(
        ir_root = '../Data/MSRS/ir',
        vis_root = '../Data/MSRS/vis',
        save_path = 'output/T2EA',
    )