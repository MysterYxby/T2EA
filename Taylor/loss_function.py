import torch 
import cv2
import numpy as np
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import torch.nn as nn
import PIL
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#loss
#高斯核初始化
g_kernel_size = 5
g_padding = 2
sigma = 3
reffac = 1
kx = cv2.getGaussianKernel(g_kernel_size,sigma)
ky = cv2.getGaussianKernel(g_kernel_size,sigma)
gaussian_kernel = np.multiply(kx,np.transpose(ky))
gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).to(device)


#梯度平方计算
def gradient(x):
    laplace = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(device)
    return F.conv2d(x, kernel, stride=1, padding=1)
    

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
        # self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        # self.weighty = nn.Parameter(data=kernely, requires_grad=False)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        gradient_B = self.sobelconv(image_B_Y)
        # gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_A, gradient_B)
        return Loss_gradient

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B):
        Loss_intensity = F.l1_loss(image_A, image_B)
        return Loss_intensity

class Taylor_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.int  = L_Intensity()
        self.grad = L_Grad()
        self.sobelconv = Sobelxy()

    def forward(self,input,result,gd):
        b,c,h,w = result.shape
        demo = torch.zeros([b,c,h,w]).to(self.device)
        loss_L1  = self.int(input,result)
        loss_g1  = self.grad(input,result)
        for i in range(len(gd)):
            demo =  torch.max(demo,gd[i])
        loss_g2 = self.int(self.sobelconv(input),demo)
        loss_g = loss_g1 + loss_g2
        return loss_L1, loss_g, loss_L1 + loss_g

if __name__ == "__main__":

    # a = torch.ones([1,1,4,4])
    # b = torch.ones([1,1,4,4])
    # c = torch.zeros([1,1,4,4])
    # loss_Function = Taylor_loss()
    # loss1,loss2,loss3 = loss_Function(a,c)
    # print(loss1)
    # print(loss2)
    # print(loss3)
    # ant =  PIL.Image.open('output/4.png')
    # ant = transforms.ToTensor()(ant)
    # ant = ant.unsqueeze(0)
    # demo = PIL.Image.open('output/4_1y.png')
    # demo = transforms.ToTensor()(demo)
    # demo = demo.unsqueeze(0)
    # sobelconv = Sobelxy()
    # g = sobelconv(demo)
    # ant_g = sobelconv(ant)
    # g = g.squeeze()
    # ant_g = ant_g.squeeze()
    # g = transforms.ToPILImage()(g)
    # ant = transforms.ToPILImage()(ant_g)
    # g.show()
    # ant.show()
    # a = torch.tensor([1,2,3])
    # print(a)
    # b = torch.tensor([2,0,5])
    # print(b)
    # c = torch.tensor([6,0,1])
    # print(c)
    # print(c.shape)
    # d = torch.zeros([3])
    # d = torch.max(d,a)
    # print(d)
    # d = torch.max(d,b)
    # print(d)
    # d = torch.max(d,c)
    # print(d)
    for i in range(len([1,2])):
        print(i)
