#train_Seg
#导入要用到的包
import torch
from torch.autograd import Variable
import time
import logging
import os.path as osp
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from network.TEM import Taylor_Encoder
from network.SegNet import BiSeNet
from loss.Task import FusionLoss,OhemCELoss
from network.FusionNet import FusionModel
from Datasets import Fusion_dataset
import random 
import numpy as np

def setup_seed(seed):
	torch.manual_seed(seed)				#为cpu分配随机种子
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)	#为gpu分配随机种子
		torch.cuda.manual_seed_all(seed)#若使用多块gpu，使用该命令设置随机种子
	random.seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

#logger
def setup_logger(logpth):
    logfile = 'BiSeNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank()==0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())




def calculate_average(num_list):
    # 检查列表是否为空
    if len(num_list) == 0:
        return 0  # 如果列表为空，则返回 0 或者可以根据需求返回其他值

    # 计算平均值
    average = sum(num_list) / len(num_list)
    return average


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
#     temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    temp = torch.cat((Y, Cr, Cb), dim=1)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
#     mat = torch.tensor(
#         [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
#     )
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    )
#     bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
#     temp = (im_flat + bias).mm(mat).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5])
    temp = (im_flat.cpu() + bias).mm(mat)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def generate_fusion(Taylor,Model,ir,image_vis_ycrcb):
     #taylor decomposition
    _,ir_y = Taylor(ir,2)
    _,vis_y = Taylor(image_vis_ycrcb[:,:1],2)

    #Fusion
    logits = Model(ir_y,vis_y)

    #to RGB
    fusion_ycrcb = torch.cat(
        (logits, image_vis_ycrcb[:, 1:2, :, :],
         image_vis_ycrcb[:, 2:, :, :]),
        dim=1,
    )

    fusion_image = YCrCb2RGB(fusion_ycrcb)
    
    return fusion_image,logits
            

def train_seg(epochs,lr,batch_size,shuffle,Taylor_path,seg_path,fusion_path):

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #模型加载---预训练完成，不用更新参数
    #//Taylor预训练模型读取，固定参数
    Taylor = Taylor_Encoder()
    Taylor.load_state_dict(torch.load(Taylor_path,map_location = torch.device(device)))
    Taylor = Taylor.to(device)
    Taylor.eval()
    for p in Taylor.parameters():
        p.requires_grad = False
    print('Load Taylor Decomposition Model Sucessfully~')

    #seg模型读取，固定参数
    n_classes = 9
    SegModel = BiSeNet(n_classes)
    SegModel.load_state_dict(torch.load(seg_path,map_location = torch.device(device)))
    SegModel = SegModel.to(device)
    SegModel.eval()
    for p in SegModel.parameters():
        p.requires_grad = False
    print('Load Segmentation Model Sucessfully~')

    #加载预训练的融合模型////不固定参数

    Model = FusionModel()
    Model.load_state_dict(torch.load(fusion_path,map_location = torch.device(device)))
    Model = Model.to(device)
    print('Load Fusion Model Sucessfully~')

    best_loss = float('inf')
    #tensorboard
    writer = SummaryWriter('logs/Seg_Fusion')

    #优化器
    optimizer= torch.optim.Adam(Model.parameters(),lr = lr)


    for epoch in range(1, epochs+1):
        num = (epoch // 10) + 1
        print("Epoch {0} / {1}".format(epoch, epochs))

        #1 训练
        #1 traiN--------------------------------------------------------------
        Model.train()
        train_dataset = Fusion_dataset('train')
        print("the training dataset is length:{}".format(train_dataset.length))
        train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        )
        train_loader.n_iter = len(train_loader)
        score_thres = 0.7
        ignore_idx = 255
#       n_min = 8 * 640 * 480 // 8
        n_min = 640 * 480 - 1 
        criteria_p = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        criteria_16 = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        criteria_fusion = FusionLoss()
        fus_l = 0
        seg_l = 0
        for it, (vis_batch, ir_batch, label, _) in enumerate(train_loader):
            #train
            ir_batch = Variable(ir_batch).to(device)
            vis_batch = Variable(vis_batch).to(device)
            image_vis_ycrcb = RGB2YCrCb(vis_batch)
            label = Variable(label).to(device)
            
            #fusion
            fusion_image,logits = generate_fusion(Taylor,Model,ir_batch,image_vis_ycrcb)
            
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            lb = torch.squeeze(label, 1)
            

            #seg
            out,mid = SegModel(fusion_image.to(device))

            #loss 
            #seg-loss
            lossp = criteria_p(out, lb)
            loss2 = criteria_16(mid, lb)
            seg_loss = lossp + 0.1 * loss2

            #fusion-loss
            _, _, loss_fusion= criteria_fusion(
                image_vis_ycrcb[:,:1], ir_batch, logits
            )
            del fusion_image, logits  # 删除不再需要的变量
            loss_total = loss_fusion + num * seg_loss
            fus_l += loss_fusion.item()/train_loader.n_iter
            seg_l += seg_loss.item()/train_loader.n_iter
            #backward
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            print('step of train: {}/{}, total:{:.4f}, fusion:{:.4f}, seg:{:.4f},'.format(epoch*it+1,train_loader.n_iter,loss_total.item(),loss_fusion.item(),num*seg_loss.item()))

        writer.add_scalar("train_fusion",fus_l,epoch)
        writer.add_scalar("train_seg",seg_l,epoch)
        print('Epoch of train: {}/{}, fusion:{:.4f}, seg:{:.4f}'.format(epoch,epochs,fus_l,seg_l))

        #2 验证
        # 2，val -------------------------------------------------  
        Model.eval()
        val_dataset = Fusion_dataset('val')
        print("the val dataset is length:{}".format(val_dataset.length))
        val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        )
        val_loader.n_iter = len(val_loader)
        #val 
        fus_l = 0
        seg_l = 0
        with torch.no_grad():
            for it, (vis_batch, ir_batch, label, _) in enumerate(val_loader):
                #val
                ir_batch = Variable(ir_batch).to(device)
                vis_batch = Variable(vis_batch).to(device)
                image_vis_ycrcb = RGB2YCrCb(vis_batch)
                label = Variable(label).to(device)
                
                #fusion
                fusion_image,logits = generate_fusion(Taylor,Model,ir_batch,image_vis_ycrcb)

    
                #seg
                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(
                    fusion_image < zeros, zeros, fusion_image)
                lb = torch.squeeze(label, 1)
                

                #seg
                out,mid = SegModel(fusion_image.to(device))

                #loss 
                #seg-loss
                lossp = criteria_p(out, lb)
                loss2 = criteria_16(mid, lb)
                seg_loss = lossp + 0.1 * loss2

                #fusion-loss
                _, _, loss_fusion= criteria_fusion(
                    image_vis_ycrcb[:,:1], ir_batch, logits
                )
                loss_total = loss_fusion + num * seg_loss
                del fusion_image, logits  # 删除不再需要的变量
                fus_l += loss_fusion.item()/train_loader.n_iter
                seg_l += seg_loss.item()/train_loader.n_iter
                # #display loss
                print('step of val: {}/{}, total:{:.4f}, fusion:{:.4f}, seg:{:.4f},'.format(epoch*it+1,val_loader.n_iter,loss_total.item(),loss_fusion.item(),num*seg_loss.item()))    
            #write loss
            writer.add_scalar("val_fusion",fus_l,epoch)
            writer.add_scalar("val_seg",seg_l,epoch)
            print('Epoch of val: {}/{}, fusion:{:.4f}, seg:{:.4f}'.format(epoch,epochs,fus_l,seg_l))
        
        if fus_l < best_loss:
            best_loss = fus_l
            #save models
            path = "./model/" + "Fusion_Seg.pt"
            torch.save(Model.state_dict(),path)
            print(f'Best model saved with test loss: {fus_l:.4f}')



if __name__ =='__main__':
    setup_seed(0)
    train_seg(epochs = 50,
                  lr = 1e-5,
                  batch_size = 2,
                  shuffle = True,
                  Taylor_path = 'model/Taylor.pt',
                  seg_path = 'model/model_final.pth',
                  fusion_path = 'model/Fusion.pt')

