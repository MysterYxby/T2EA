'''
for training Taylor
'''
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter   
import torch
from loss.Taylor import Taylor_loss
from network.TEM import Taylor_Encoder
from Datasets import TaylorDataset
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

def train_Taylor_Encoder(train_root,test_root,batch_size,shuffle,epochs,Layer,lr):

    #准备环境--待训练模型--损失函数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_ft = Taylor_loss()
    Model = Taylor_Encoder().to(device)
    best_loss = float('inf')
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
        dataset_train = TaylorDataset(train_root)
        train_loader = DataLoader(dataset_train,batch_size = batch_size,shuffle = shuffle, num_workers = 0)
        for _,(img_batch) in enumerate (train_loader):
            #predict
            img = img_batch.to(device)
            out, y = Model(img,Layer)
            # print(out[0])

            #loss
            loss_int, loss_grad, loss = loss_ft(img,out,y)
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
        writer.add_scalar("train_int_loss",train_int,epoch)
        writer.add_scalar("train_grad_loss",train_grad,epoch)

        #2 验证
        # 2，val -------------------------------------------------  
        Model.eval()
        val_loss = 0
        val_int = 0
        val_grad = 0
        dataset_val = TaylorDataset(test_root)
        val_loader = DataLoader(dataset_val, batch_size = batch_size, shuffle = shuffle, num_workers = 0)
        with torch.no_grad():
            for _,(img_batch) in enumerate (train_loader):
                #predict
                img = img_batch.to(device)
                out, y = Model(img,Layer)
           
                #loss
                loss_int, loss_grad, loss = loss_ft(img,out,y)
                val_int += loss_int.item()/len(val_loader.dataset)
                val_grad += loss_grad.item()/len(val_loader.dataset)
                val_loss += loss.item()/len(val_loader.dataset)

        #display loss 
        print("test: all:{:.4f}, int:{:.4f}, grad:{:.4f},".format(val_loss,val_int,val_grad))
        writer.add_scalar("val_loss",val_loss,epoch)
        writer.add_scalar("val_int_loss",val_int,epoch)
        writer.add_scalar("val_grad_loss",val_grad,epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            #save models
            path = "./model/" + "Taylor.pt"
            torch.save(Model.state_dict(),path)
            print(f'Best model saved with test loss: {val_loss:.4f}')
        


if __name__ =='__main__':
    setup_seed(0)
    train_Taylor_Encoder(train_root = '../Data/train',
                     test_root  =  '../Data/test',
                     batch_size = 2,
                     shuffle = True,
                     epochs = 200,
                     Layer = 2,
                     lr = 1e-5)