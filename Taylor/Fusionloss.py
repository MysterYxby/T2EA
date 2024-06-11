#loss Function
import torch.nn as nn 
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B,image_F):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_F_Y = image_F[:, :1, :, :]
        gradient_F = self.sobelconv(image_F_Y)
        gradient_A = self.sobelconv(image_A_Y)
        gradient_B = self.sobelconv(image_B_Y)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_F, gradient_joint)
        return Loss_gradient

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B,image_F):
        Loss_intensity = F.l1_loss(image_F, torch.max(image_A, image_B))
        return Loss_intensity
class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.int = L_Intensity()
        self.grad = L_Grad()
        
    def forward(self, image_A,image_B,image_F):
        loss_int = self.int(image_A,image_B,image_F)
        loss_grad = self.grad(image_A,image_B,image_F)
        loss = loss_int + 10*loss_grad
        return loss_int, loss_grad, loss
    
