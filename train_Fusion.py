'''
For training the fusion network
plz load the pre-trianed Taylor model
'''

import torch
from network.TEM import Taylor_Encoder
from network.FusionNet import FusionModel
from loss.Fusion import Fusionloss
from Datasets import FusionDataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import random

def setup_seed(seed):
	torch.manual_seed(seed)				#为cpu分配随机种子
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)	#为gpu分配随机种子
		torch.cuda.manual_seed_all(seed)#若使用多块gpu，使用该命令设置随机种子
	random.seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
     

def train_fusion(epochs,lr,batch_size,shuffle,train_ir_path,train_vis_path,test_ir_path,test_vis_path,Taylor_path):

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #模型加载
    #//Taylor预训练模型读取，固定参数
    Net = Taylor_Encoder()
    Net.load_state_dict(torch.load(Taylor_path))
    Net = Net.to(device)
    
    #定义待训练融合模型
    Model = FusionModel()
    Model = Model.to(device)
    
    #损失函数
    loss_ft = Fusionloss() #A,B,F
    best_loss = float('inf')
    #tensorboard
    writer = SummaryWriter('logs/Fusion')

    #优化器
    optimizer= torch.optim.Adam(Model.parameters(),lr = lr)

    for epoch in range(1, epochs+1):

        print("Epoch {0} / {1}".format(epoch, epochs))
        train_int = 0
        train_grad = 0
        train_loss = 0

        #1 训练
        Model.train()
        dataset_train = FusionDataset(train_ir_path,train_vis_path)
        train_loader = DataLoader(dataset_train,batch_size = batch_size,shuffle = shuffle, num_workers = 0)
        for _,(ir_batch,vis_batch) in enumerate (train_loader):

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
            train_int += loss_int.item()/len(train_loader.dataset)
            train_grad += loss_grad.item()/len(train_loader.dataset)
            train_loss += loss.item()/len(train_loader.dataset)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #display loss 
        # print("train: all:{:.4f}".format(train_loss/train_batch))
        print("train: all:{:.4f}, int:{:.4f}, grad:{:.4f},".format(train_loss,train_int,train_grad))
        writer.add_scalar("train_loss",train_loss,epoch)
        writer.add_scalar("train_int_loss",loss_int,epoch)
        writer.add_scalar("train_grad_loss",loss_grad,epoch)

        #2 验证
        # 2，val -------------------------------------------------  
        Model.eval()
        val_loss = 0
        val_int = 0
        val_grad = 0
        dataset_val = FusionDataset(test_ir_path,test_vis_path)
        val_loader = DataLoader(dataset_val, batch_size = batch_size, shuffle = shuffle, num_workers = 0)
        with torch.no_grad():
            for _,(ir_batch,vis_batch) in enumerate (val_loader):

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
                val_int += loss_int.item()/len(val_loader.dataset)
                val_grad += loss_grad.item()/len(val_loader.dataset)
                train_loss += loss.item()/len(val_loader.dataset)
                val_loss += loss/len(val_loader.dataset)

        #display loss 
        print("test: all:{:.4f}, int:{:.4f}, grad:{:.4f},".format(val_loss,val_int,val_grad))
        writer.add_scalar("val_loss",val_loss,epoch)
        writer.add_scalar("val_int_loss",val_int,epoch)
        writer.add_scalar("val_grad_loss",val_grad,epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            #save models
            path = "./model/" + "Fusion.pt"
            torch.save(Model.state_dict(),path)
            print(f'Best model saved with test loss: {val_loss:.4f}')

if __name__ == '__main__':
    setup_seed(0)
    train_fusion(epochs = 200,
             lr = 1e-5,
             batch_size = 2,
             shuffle = True,
             train_ir_path = '../../IVFusion/dataset/MSRS/Infrared/train',
             train_vis_path = '../../IVFusion/dataset/MSRS/Visible/train',
             test_ir_path = '../../IVFusion/dataset/MSRS/Infrared/test',
             test_vis_path = '../../IVFusion/dataset/MSRS/Visible/test',
             Taylor_path = 'model//Taylor.pt')


