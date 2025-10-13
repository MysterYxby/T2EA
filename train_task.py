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
    device = input_im.device
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
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
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
    device = input_im.device
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]],
        device=device
    )
    bias = torch.tensor([0.0 / 255, -0.5, -0.5], device=device)
    temp = (im_flat + bias).mm(mat)
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

def generate_fusion(Taylor, Model, ir, image_vis_ycrcb):
    # 确保输入在正确的设备上
    ir = ir.to(next(Taylor.parameters()).device)
    image_vis_ycrcb = image_vis_ycrcb.to(next(Taylor.parameters()).device)
    
    #taylor decomposition
    with torch.no_grad():
        _, ir_y = Taylor(ir, 2)
        _, vis_y = Taylor(image_vis_ycrcb[:, :1], 2)

    #Fusion
    logits = Model(ir_y, vis_y)

    #to RGB
    fusion_ycrcb = torch.cat(
        (logits, image_vis_ycrcb[:, 1:2, :, :],
         image_vis_ycrcb[:, 2:, :, :]),
        dim=1,
    )

    fusion_image = YCrCb2RGB(fusion_ycrcb)
    
    return fusion_image, logits

def train_seg(epochs, lr, batch_size, shuffle, Taylor_path, seg_path, fusion_path):
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #模型加载---预训练完成，不用更新参数
    #Taylor预训练模型读取，固定参数
    Taylor = Taylor_Encoder()
    try:
        Taylor.load_state_dict(torch.load(Taylor_path, map_location=device))
        Taylor = Taylor.to(device)
        Taylor.eval()
        for p in Taylor.parameters():
            p.requires_grad = False
        print('Load Taylor Decomposition Model Sucessfully~')
    except Exception as e:
        print(f'Error loading Taylor model: {e}')
        return

    #seg模型读取，固定参数
    n_classes = 9
    SegModel = BiSeNet(n_classes)
    try:
        SegModel.load_state_dict(torch.load(seg_path, map_location=device))
        SegModel = SegModel.to(device)
        SegModel.eval()
        for p in SegModel.parameters():
            p.requires_grad = False
        print('Load Segmentation Model Sucessfully~')
    except Exception as e:
        print(f'Error loading Segmentation model: {e}')
        return

    #加载预训练的融合模型////不固定参数
    Model = FusionModel()
    try:
        Model.load_state_dict(torch.load(fusion_path, map_location=device))
        Model = Model.to(device)
        print('Load Fusion Model Sucessfully~')
    except Exception as e:
        print(f'Error loading Fusion model: {e}')
        # 如果加载失败，使用初始化的模型
        Model = Model.to(device)

    best_loss = float('inf')
    #tensorboard
    writer = SummaryWriter('logs/Seg_Fusion')

    #优化器
    optimizer = torch.optim.Adam(Model.parameters(), lr=lr)

    # 损失函数
    score_thres = 0.7
    ignore_idx = 255
    n_min = 640 * 480 - 1 
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_fusion = FusionLoss()

    for epoch in range(1, epochs+1):
        num = (epoch // 10) + 1
        print("Epoch {0} / {1}".format(epoch, epochs))

        # 1 训练
        Model.train()
        train_dataset = Fusion_dataset('train')
        print("the training dataset is length:{}".format(len(train_dataset)))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        train_loader.n_iter = len(train_loader)
        
        fus_l_train = 0
        seg_l_train = 0
        
        for it, (vis_batch, ir_batch, label, _) in enumerate(train_loader):
            # 将数据移动到设备
            ir_batch = ir_batch.to(device)
            vis_batch = vis_batch.to(device)
            label = label.to(device)
            
            image_vis_ycrcb = RGB2YCrCb(vis_batch)
            
            #fusion
            fusion_image, logits = generate_fusion(Taylor, Model, ir_batch, image_vis_ycrcb)
            
            # 限制fusion_image在[0,1]范围内
            fusion_image = torch.clamp(fusion_image, 0, 1)
            
            #seg
            lb = torch.squeeze(label, 1)
            out, mid = SegModel(fusion_image)

            #loss 
            #seg-loss
            lossp = criteria_p(out, lb)
            loss2 = criteria_16(mid, lb)
            seg_loss = lossp + 0.1 * loss2

            #fusion-loss
            _, _, loss_fusion = criteria_fusion(
                image_vis_ycrcb[:, :1], ir_batch, logits
            )
            
            loss_total = loss_fusion + num * seg_loss
            fus_l_train += loss_fusion.item()
            seg_l_train += seg_loss.item()
            
            #backward
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            if (it + 1) % 10 == 0:  # 每10个batch打印一次
                print('Train - Epoch: {}/{}, Step: {}/{}, total:{:.4f}, fusion:{:.4f}, seg:{:.4f}'.format(
                    epoch, epochs, it+1, train_loader.n_iter, loss_total.item(), 
                    loss_fusion.item(), num*seg_loss.item()))

        # 计算平均训练损失
        avg_fus_l_train = fus_l_train / len(train_loader)
        avg_seg_l_train = seg_l_train / len(train_loader)
        
        writer.add_scalar("train_fusion", avg_fus_l_train, epoch)
        writer.add_scalar("train_seg", avg_seg_l_train, epoch)
        print('Train - Epoch: {}/{}, fusion:{:.4f}, seg:{:.4f}'.format(
            epoch, epochs, avg_fus_l_train, avg_seg_l_train))

        # 2 验证
        Model.eval()
        val_dataset = Fusion_dataset('val')
        print("the val dataset is length:{}".format(len(val_dataset)))
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证集通常不需要shuffle
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        
        fus_l_val = 0
        seg_l_val = 0
        
        with torch.no_grad():
            for it, (vis_batch, ir_batch, label, _) in enumerate(val_loader):
                # 将数据移动到设备
                ir_batch = ir_batch.to(device)
                vis_batch = vis_batch.to(device)
                label = label.to(device)
                
                image_vis_ycrcb = RGB2YCrCb(vis_batch)
                
                #fusion
                fusion_image, logits = generate_fusion(Taylor, Model, ir_batch, image_vis_ycrcb)
                
                # 限制fusion_image在[0,1]范围内
                fusion_image = torch.clamp(fusion_image, 0, 1)
                
                #seg
                lb = torch.squeeze(label, 1)
                out, mid = SegModel(fusion_image)

                #loss 
                #seg-loss
                lossp = criteria_p(out, lb)
                loss2 = criteria_16(mid, lb)
                seg_loss = lossp + 0.1 * loss2

                #fusion-loss
                _, _, loss_fusion = criteria_fusion(
                    image_vis_ycrcb[:, :1], ir_batch, logits
                )
                
                loss_total = loss_fusion + num * seg_loss
                fus_l_val += loss_fusion.item()
                seg_l_val += seg_loss.item()
                
                if (it + 1) % 10 == 0:  # 每10个batch打印一次
                    print('Val - Epoch: {}/{}, Step: {}/{}, total:{:.4f}, fusion:{:.4f}, seg:{:.4f}'.format(
                        epoch, epochs, it+1, len(val_loader), loss_total.item(), 
                        loss_fusion.item(), num*seg_loss.item()))
        
        # 计算平均验证损失
        avg_fus_l_val = fus_l_val / len(val_loader)
        avg_seg_l_val = seg_l_val / len(val_loader)
        
        writer.add_scalar("val_fusion", avg_fus_l_val, epoch)
        writer.add_scalar("val_seg", avg_seg_l_val, epoch)
        print('Val - Epoch: {}/{}, fusion:{:.4f}, seg:{:.4f}'.format(
            epoch, epochs, avg_fus_l_val, avg_seg_l_val))
        
        # 保存最佳模型
        if avg_fus_l_val < best_loss:
            best_loss = avg_fus_l_val
            # 确保model目录存在
            import os
            os.makedirs("./model", exist_ok=True)
            path = "./model/Fusion_Seg.pt"
            torch.save(Model.state_dict(), path)
            print(f'Best model saved with val fusion loss: {avg_fus_l_val:.4f}')

if __name__ =='__main__':
    setup_seed(0)
    train_seg(epochs=50,
              lr=1e-5,
              batch_size=2,
              shuffle=True,
              Taylor_path='model/Taylor.pt',
              seg_path='model/model_final.pth',
              fusion_path='model/Fusion.pt')