import torch
from torch import nn
import torch.nn.functional as F
from math import factorial


#残差模块
class ResB(nn.Module):
    def __init__(self,in_c,out_c):
        super(ResB,self).__init__()
        self.Res = nn.Sequential(
            nn.Conv2d(in_c,out_c,1,1,0),
            nn.LeakyReLU(),
            nn.Conv2d(out_c,out_c,3,1,1),
            nn.LeakyReLU(),
            nn.Conv2d(out_c,out_c,1,1,0),
        )
        self.Conv = nn.Conv2d(in_c,out_c,1,1,0)
        self.activate = nn.LeakyReLU()
    def forward(self, x):
        y = self.Res(x) + self.Conv(x)
        y = self.activate(y)
        return y


class Conv(nn.Module):
    def __init__(self,in_c,o_c,size,stride):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=o_c,kernel_size=size, stride=stride, padding=size//2)
        self.activation = nn.Mish()

    def forward(self,input):
        x = self.conv(input)
        y = self.activation(x)
        return y


#残差密集连接块
class RDBLOCK(nn.Module):
    def __init__(self,outchannel):  #  x的outchannnel
        super(RDBLOCK, self).__init__()
        self.conv1 = Conv(in_c=outchannel,o_c=outchannel,size=1,stride=1)   #xoutchannel作为x1的inchannel
        self.conv2 = Conv(in_c=2*outchannel,o_c=outchannel,size=3,stride=1)   #x2的inc为 x和x1的oc相加
        self.conv3 = Conv(in_c=3*outchannel,o_c=outchannel,size=1,stride=1)
        self.conv4 = Conv(in_c=4*outchannel,o_c=2*outchannel,size=1,stride=1)
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels=outchannel,out_channels=2*outchannel,kernel_size=1,stride=1))

    def forward(self,x):
        x1 = self.conv1(x)
        y = torch.cat((x,x1),1)
        x2 = self.conv2(y)
        z = torch.cat((x,x1,x2),1)
        x3 = self.conv3(z)
        x4 = self.conv4(torch.cat((z,x3),1))
        output = self.shortcut(x) + x4
        output = F.mish(output)
        return output


class Taylor_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(1,32,5,1,2),
            nn.LeakyReLU(),
            ResB(32,64),
            ResB(64,32),
            ResB(32,1),
        )

        self.gradient = nn.Sequential(
            nn.Conv2d(2,8,5,1,2),
            nn.LeakyReLU(),
            RDBLOCK(8),
            RDBLOCK(16),
            RDBLOCK(32),
            nn.Conv2d(64,1,5,1,2),
            nn.LeakyReLU(),
        )

    def forward(self,input,n):
        y_list = []
        x = self.base(input)  #semantic/base分支
        y_list.append(x)
        b,c,h,w = x.shape
        result = torch.zeros([b,c,h,w])
        for i in range(1,n+1):
            y_list.append(self.gradient(torch.cat([y_list[i-1],input],dim=1))) 
        for i in range(len(y_list)):
            result += (1 / factorial(i) ) * y_list[i]
        return  result, y_list
    

if __name__ == '__main__':
    test_img = torch.normal(0,1,(1,1,256,256))
    print(test_img.shape)
    model = Taylor_Encoder()
    img, y = model(test_img,2)
    print(img.shape)
    print(len(y))
