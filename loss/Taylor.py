#loss
import torch.nn.functional as F
import torch
import torch.nn as nn


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
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(self.device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(self.device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        gradient_B = self.sobelconv(image_B_Y)
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
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def forward(self,input,result,gd):
        b,c,h,w = result.shape
        grad_map = torch.zeros([b,c,h,w]).to(self.device)
        loss_L1  = self.int(input,result)
        loss_g1  = self.grad(input,result)
        for i in range(1,len(gd)):
            grad_map =  torch.max(grad_map,gd[i])
        loss_g2 = self.int(self.sobelconv(input),grad_map)
        loss_g = loss_g1 + 0.3 * loss_g2
        return loss_L1, loss_g, loss_L1 + loss_g