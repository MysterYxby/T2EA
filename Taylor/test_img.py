import torch
from TEM import Taylor_Encoder
from PIL import Image
from utils import tensor_rgb2ycbcr,tensor_ycbcr2rgb,save_PIL,transform_img,save_seg
import os
from torchvision import transforms
from FusionNetwork import FusionModel
from Seg_Net import BiSeNet
from math import factorial


def fusion_test(vis,ir,c,save_path,Net,Fusion,device,filenames_ir,filenames_vis,i):
    with torch.no_grad():
        
        if c == 3:
            #融合图像应该为彩色
            #img1, img1_cb, img1_cr = tensor_rgb2ycbcr(img)
            gray, img1_cb, img1_cr = tensor_rgb2ycbcr(vis)

            #Taylor Encoder
            _,y_vis = Net(gray.to(device),2)
            _,y_ir  = Net(ir.to(device),2)
            # b,_,h,w = y_vis[0].shape
            # src = torch.zeros([b,1,h,w])
            # for n in range(len(y_vis)):
            #     src += (1/factorial(n)) * y_vis[n]
            #     save_PIL(tensor_ycbcr2rgb(src, img1_cb.to(device), img1_cr.to(device)),os.path.join('output/Taylor/final', 'va_' + str(n) + '_'+ filenames_vis[i]))
            # for n in range(len(y_ir)):
            #     save_PIL(y_ir[n],os.path.join('output/Taylor/final', 'r_' + str(n) + '_'+ filenames_vis[i]))   
            # # Fusion
            result = Fusion(y_vis,y_ir)

            #颜色恢复
            result =  tensor_ycbcr2rgb(result, img1_cb.to(device), img1_cr.to(device))
            # src_1 = torch.zeros([b,c,h,w])
            # for n in range(len(y_vis)):
            #     src_1 += (1/factorial(n)) * f_list[n]
            #     save_PIL(tensor_ycbcr2rgb(f_list[n], img1_cb.to(device), img1_cr.to(device)),os.path.join('output/Taylor/final', 'fa_' + str(n) + '_'+ filenames_vis[i]))
            # # 保存融合图像
            save_PIL(result,os.path.join(save_path,filenames_vis[i]))
            print('the {}th pair of image is  Ok!'.format(str(i+1)))
        else:
            #Taylor Encoder
            _,y_vis = Net(vis.to(device),2)
            _,y_ir  = Net(ir.to(device),2)
            # b,_,h,w = y_vis[0].shape
            # src = torch.zeros([b,1,h,w])
            # for n in range(len(y_vis)):
            #     src += (1/factorial(n)) * y_ir[n]
                
            #     save_PIL(transforms.Resize([480,640])(src), os.path.join('output/Taylor/final', 'ra_' + str(n) + '_'+ filenames_vis[i]))
            # # for n in range(len(y_ir)):
            #     save_PIL(transforms.Resize([480,640])(y_ir[n]),os.path.join('output/Taylor/final', 'r_' + str(n) + '_'+ filenames_vis[i]))  
            # #Fusion
            result = Fusion(y_vis,y_ir)
            
            # src_1 = torch.zeros([b,c,h,w])
            # for n in range(len(y_vis)):
            #     src_1 += (1/factorial(n)) * f_list[n]
            #     save_PIL(transforms.Resize([480,640])(f_list[n]),os.path.join('output/Taylor/final', 'fa_' + str(n) + '_'+ filenames_vis[i]))
            #保存融合图像
            save_PIL(result,os.path.join(save_path,filenames_vis[i]))  
            print('the {}th pair of image is  Ok!'.format(str(i+1)))
        
    return 0
    

def test_Fusion(ir_root,vis_root,model_fusion,model_Taylor,save_path):
     #判断保存图像路径是否存在
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #模型加载
    Net = Taylor_Encoder()
    Net.load_state_dict(torch.load(model_Taylor, map_location=torch.device('cpu')))
    Net = Net.to(device)
    Net = Net.eval()
    print('Taylor Decomposition is successfully loaded!')
    Fusion = FusionModel()
    Fusion.load_state_dict(torch.load(model_fusion, map_location=torch.device('cpu')))
    Fusion = Fusion.to(device)
    Fusion = Fusion.eval()
    print('Taylor Fusion is successfully loaded!')
    
    #读取文件
    filenames_ir = os.listdir(ir_root)
    filenames_vis = os.listdir(vis_root)
    print('The number of total images is {} pairs'.format(str(len(filenames_ir))))
    for i in range (len(filenames_ir)):
        filepath_ir = os.path.join(ir_root,filenames_ir[i])
        ir = Image.open(filepath_ir).convert('L')
        ir = transform_img(ir)
        filepath_vis = os.path.join(vis_root,filenames_vis[i])
        vis = Image.open(filepath_vis).convert('L')
        vis = transform_img(vis)
        _,c,_,_ = vis.size()
        fusion_test(vis,ir,c,save_path,Net,Fusion,device,filenames_ir,filenames_vis,i) 

def proc_seg(model,img,save_path,filenames):
    seg_map,_ = model(img)
    save_seg(seg_map,os.path.join(save_path,filenames))
    
    

def test_Seg(root,save_path,model_path):
    #判断保存图像路径是否存在
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #模型加载
    n_classes = 9
    Seg = BiSeNet(n_classes)
    Seg.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    Seg.eval()
    for p in Seg.parameters():
        p.requires_grad = False
    print('Load Segmentation Model Sucessfully~')

    #读取数据
    filenames = os.listdir(root)

    print('The number of total images is {} '.format(str(len(filenames))))

    for i in range (len(filenames)):
        filepath = os.path.join(root,filenames[i])
        img = Image.open(filepath)
        img = transform_img(img)
        
        #segmentation 
        proc_seg(Seg,img,save_path,filenames[i])

    return 0


if __name__ == '__main__':
    pass
    # please choose the relevant function to run for test the procedures

    # test_Fusion(
    #     ir_root = 'data/ir',
    #     vis_root = 'data/vis',
    #     model_fusion = '/model/FusionNet.pt',
    #     model_Taylor = 'model/Taylor.pt',
    #     save_path = 'output/Fusion/'
    # )

    # test_Seg(
    #     root  = 'output/Fusion/' ,
    #     save_path = 'output/Segmentation/',
    #     model_path = 'code/model/Fusion/model_final.pth'
    # )
