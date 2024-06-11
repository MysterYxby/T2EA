from TEM import Taylor_Encoder
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from Dataloader import MyDataset_OneImg, MyDataset, Fusion_dataset
import torch
from loss_function import Taylor_loss  
from torch.autograd import Variable
from Seg_Net import BiSeNet
# from TaskFusion_dataset import Fusion_dataset
from loss import OhemCELoss
import argparse
from Fusionloss import FusionLoss
from FusionNetwork import FusionModel
from Seg_Net import BiSeNet

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

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
    # temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
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
    # mat = torch.tensor(
    #     [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    # ).cuda()
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    )
    # bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5])
    # temp = (im_flat + bias).mm(mat).cuda()
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

def train_Taylor_Encoder(train_root,test_root,batch_size,shuffle,epochs):

    #准备环境--待训练模型--损失函数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_ft = Taylor_loss()
    Model = Taylor_Encoder().to(device)

    #tensorboard
    writer = SummaryWriter('logs/Taylor')
    
    #优化器
    optimizer= torch.optim.Adam(Model.parameters(),lr = 0.0001)

    for epoch in range(1, epochs+1):

        print("Epoch {0} / {1}".format(epoch, epochs))
        train_int = 0
        train_grad = 0
        train_loss = 0

        #1 训练
        Model.train()
        dataset_train = MyDataset_OneImg(train_root)
        train_loader = DataLoader(dataset_train,batch_size = batch_size,shuffle = shuffle, num_workers = 0)
        train_batch = 0
        for _,(img_batch) in enumerate (train_loader):
            train_batch += 1 
            #predict
            img = img_batch.to(device)
            out, _, _ = Model(img,2)
            # print(out[0])

            #loss
            loss_int, loss_grad, loss = loss_ft(img,out)
            train_int += loss_int.item()/batch_size
            train_grad += loss_grad.item()/batch_size
            train_loss += loss.item()/batch_size
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #display loss 
        # print("train: all:{:.4f}".format(train_loss/train_batch))
        print("train: all:{:.4f}, int:{:.4f}, grad:{:.4f},".format(train_loss/train_batch,loss_int/train_batch,loss_grad/train_batch))
        writer.add_scalar("train_loss",train_loss/train_batch,epoch)
        writer.add_scalar("train_int_loss",loss_int/train_batch,epoch)
        writer.add_scalar("train_grad_loss",loss_grad/train_batch,epoch)

        #2 验证
        # 2，val -------------------------------------------------  
        Model.eval()
        val_loss = 0
        val_int = 0
        val_grad = 0
        dataset_val = MyDataset_OneImg(test_root)
        train_loader = DataLoader(dataset_val, batch_size = batch_size, shuffle = shuffle, num_workers = 0)
        test_batch = 0
        with torch.no_grad():
            for _,(img_batch) in enumerate (train_loader):
                test_batch += 1 
                #predict
                img = img_batch.to(device)
                out, _, _ = Model(img,2)
           
                #loss
                val_int += loss_int.item()/batch_size
                val_grad += loss_grad.item()/batch_size
                train_loss += loss.item()/batch_size
                val_loss += loss/batch_size

        #display loss 
        print("test: all:{:.4f}, int:{:.4f}, grad:{:.4f},".format(val_loss/test_batch,loss_int/test_batch,loss_grad/test_batch))
        writer.add_scalar("val_loss",val_loss/test_batch,epoch)
        writer.add_scalar("val_int_loss",val_int/test_batch,epoch)
        writer.add_scalar("val_grad_loss",val_grad/test_batch,epoch)

        if epoch >=1800:
        #save models
            path = "./model/" + "model{}.pt".format(epoch)
            torch.save(Model.state_dict(),path)


def train_fusion(epochs,lr,batch_size,shuffle,train_ir_path,train_vis_path,test_ir_path,test_vis_path,Taylor_path):

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #模型加载
    #//Taylor预训练模型读取，固定参数
    Net = Taylor_Encoder()
    Net.load_state_dict(torch.load(Taylor_path,map_location = torch.device(device)))

    #定义待训练融合模型
    Model = FusionModel()

    #损失函数
    loss_ft = FusionLoss() #A,B,F

    #tensorboard
    writer = SummaryWriter('logs/Taylor')

    #优化器
    optimizer= torch.optim.Adam(Model.parameters(),lr = lr)

    for epoch in range(1, epochs+1):

        print("Epoch {0} / {1}".format(epoch, epochs))
        train_int = 0
        train_grad = 0
        train_loss = 0

        #1 训练
        Model.train()
        dataset_train = MyDataset(train_ir_path,train_vis_path)
        train_loader = DataLoader(dataset_train,batch_size = batch_size,shuffle = shuffle, num_workers = 0)
        train_batch = 0
        for _,(ir_batch,vis_batch) in enumerate (train_loader):
            train_batch += 1 

            #train
            ir = ir_batch.to(device)
            vis = vis_batch.to(device)

            #taylor decomposition
            _,ir_y = Net(ir,2)
            _,vis_y = Net(vis,2)

            #Fusion
            result = Model(ir_y,vis_y)

            #loss
            loss_int, loss_grad, loss = loss_ft(ir,vis,result)
            train_int += loss_int.item()/batch_size
            train_grad += loss_grad.item()/batch_size
            train_loss += loss.item()/batch_size

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #display loss 
        # print("train: all:{:.4f}".format(train_loss/train_batch))
        print("train: all:{:.4f}, int:{:.4f}, grad:{:.4f},".format(train_loss/train_batch,loss_int/train_batch,loss_grad/train_batch))
        writer.add_scalar("train_loss",train_loss/train_batch,epoch)
        writer.add_scalar("train_int_loss",loss_int/train_batch,epoch)
        writer.add_scalar("train_grad_loss",loss_grad/train_batch,epoch)

        #2 验证
        # 2，val -------------------------------------------------  
        Model.eval()
        val_loss = 0
        val_int = 0
        val_grad = 0
        dataset_val = MyDataset(test_ir_path,test_vis_path)
        train_loader = DataLoader(dataset_val, batch_size = batch_size, shuffle = shuffle, num_workers = 0)
        test_batch = 0
        with torch.no_grad():
            for _,(ir_batch,vis_batch) in enumerate (train_loader):
                test_batch += 1 

                #test
                ir = ir_batch.to(device)
                vis = vis_batch.to(device)

                #taylor decomposition
                _,ir_y = Net(ir,2)
                _,vis_y = Net(vis,2)

                #Fusion
                result = Model(ir_y,vis_y)
           
                #loss
                loss_int, loss_grad, loss = loss_ft(ir,vis,result)
                val_int += loss_int.item()/batch_size
                val_grad += loss_grad.item()/batch_size
                train_loss += loss.item()/batch_size
                val_loss += loss/batch_size

        #display loss 
        print("test: all:{:.4f}, int:{:.4f}, grad:{:.4f},".format(val_loss/test_batch,loss_int/test_batch,loss_grad/test_batch))
        writer.add_scalar("val_loss",val_loss/test_batch,epoch)
        writer.add_scalar("val_int_loss",val_int/test_batch,epoch)
        writer.add_scalar("val_grad_loss",val_grad/test_batch,epoch)

        #save models
        path = "./" + "model{}.pt".format(epoch)
        torch.save(Model.state_dict(),path)


    return 0

def train_seg(epochs,lr,batch_size,shuffle,Taylor_path,seg_path,fusion_path):

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #模型加载---预训练完成，不用更新参数
    #//Taylor预训练模型读取，固定参数
    Taylor = Taylor_Encoder()
    Taylor.load_state_dict(torch.load(Taylor_path,map_location = torch.device(device)))
    Taylor = Taylor.to(device)
    Taylor.eval()

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


    #tensorboard
    writer = SummaryWriter('logs/Seg_Fusion')

    #优化器
    optimizer= torch.optim.Adam(Model.parameters(),lr = lr)


    for epoch in range(1, epochs+1):
        if(epoch > 50):
            num = 4
        else:
            num = 2
        print("Epoch {0} / {1}".format(epoch, epochs))

        #1 训练
        #1 trai--------------------------------------------------------------
        Model.train()
        train_dataset = Fusion_dataset('train')
        print("the training dataset is length:{}".format(train_dataset.length))
        train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
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

        for it, (vis_batch, ir_batch, label, _) in enumerate(train_loader):
            #train
            ir_batch = Variable(ir_batch).to(device)
            vis_batch = Variable(vis_batch).to(device)
            image_vis_ycrcb = RGB2YCrCb(vis_batch)
            #taylor decomposition
            _,ir_y = Taylor(ir_batch,2)
            _,vis_y = Taylor(image_vis_ycrcb[:,:1],2)

            #Fusion
            logits = Model(ir_y,vis_y)

            #to RGB
            label = Variable(label).to(device)
            
            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )

            fusion_image = YCrCb2RGB(fusion_ycrcb)
            
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            lb = torch.squeeze(label, 1)
            

            #seg
            out,mid = SegModel(fusion_image)

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

            #backward
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            print('step of train: {}/{}, total:{:.4f}, fusion:{:.4f}, seg:{:.4f},'.format(train_loader.n_iter * epoch + it,train_loader.n_iter * epochs,loss_total,loss_fusion,num*seg_loss))

        writer.add_scalar("train_total",loss_total,epoch)
        writer.add_scalar("train_fusion",loss_fusion,epoch)
        writer.add_scalar("train_seg",seg_loss,epoch)

        #2 验证
        # 2，val -------------------------------------------------  
        Model.eval()
        val_dataset = Fusion_dataset('val')
        print("the val dataset is length:{}".format(val_dataset.length))
        val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        )
        val_loader.n_iter = len(val_loader)
        #val 
        with torch.no_grad():
            for it, (vis_batch, ir_batch, label, _) in enumerate(val_loader):
                #val
                ir_batch = Variable(ir_batch).to(device)
                vis_batch = Variable(vis_batch).to(device)
                image_vis_ycrcb = RGB2YCrCb(vis_batch)
                #taylor decomposition
                _,ir_y = Taylor(ir_batch,2)
                _,vis_y = Taylor(image_vis_ycrcb[:,:1],2)

                #Fusion
                logits = Model(ir_y,vis_y)

                #to RGB
                label = Variable(label).to(device)
                
                fusion_ycrcb = torch.cat(
                    (logits, image_vis_ycrcb[:, 1:2, :, :],
                    image_vis_ycrcb[:, 2:, :, :]),
                    dim=1,
                )

                fusion_image = YCrCb2RGB(fusion_ycrcb)
                
                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(
                    fusion_image < zeros, zeros, fusion_image)
                lb = torch.squeeze(label, 1)
                

                #seg
                out,mid = SegModel(fusion_image)

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

                # #display loss
                print('step of train: {}/{}, total:{:.4f}, fusion:{:.4f}, seg:{:.4f},'.format(val_loader.n_iter * epoch + it,train_loader.n_iter * epochs,loss_total,loss_fusion,num*seg_loss))    
            #write loss
            writer.add_scalar("val_total",loss_total,epoch)
            writer.add_scalar("val_fusion",loss_fusion,epoch)
            writer.add_scalar("val_seg",seg_loss,epoch)
        
        # #save model
        path = "./" + "model{}.pt".format(epoch)
        torch.save(Model.state_dict(),path)
    return 0



if __name__ == "__main__":
    
    pass