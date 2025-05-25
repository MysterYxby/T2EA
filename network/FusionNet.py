# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import factorial

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class FusionNetwork(nn.Module):
    def __init__(self, output):
        super(FusionNetwork, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=RGBD(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = RGBD(vis_ch[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = RGBD(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = RGBD(inf_ch[1], inf_ch[2])
        #attention
        self.attention = cbam_block(vis_ch[2]+inf_ch[2])

        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2]+inf_ch[2], vis_ch[1]+vis_ch[1],padding = 3,dilation = 3)
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0],padding = 3,dilation = 3)
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0],padding = 3,dilation = 3)
        self.decode1 = ConvBnTanh2d(vis_ch[0], output) 

    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)

        #attention
        x_attention =  self.attention(torch.cat((x_vis_p2,x_inf_p2),dim=1))

        # decode
        x=self.decode4(x_attention)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x

class FusionModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Net = FusionNetwork(output=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, ir, vis):
        f_list = []
        n = len(ir)
        b,c,h,w = ir[0].shape
        result = torch.zeros([b,c,h,w]).to(self.device)
        for i in range(n):
            f_list.append(self.Net(ir[i],vis[i]))
            result += (1/factorial(i)) * f_list[i].to(self.device)
        return result
 



def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(1,1,480,640).astype(np.float32))
    y = torch.tensor(np.random.rand(1,1,480,640).astype(np.float32))
    list_x = []
    list_y = []
    list_x.append(x)
    list_y.append(y)
    model = FusionModel()
    z = model(list_x,list_y)
    print('output shape:', z.shape)
    print('test ok!')

if __name__ == '__main__':
    unit_test()