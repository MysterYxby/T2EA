#loss
import torch
import torch.nn as nn 
import torch.nn.functional as F

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
        # gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_A, gradient_B)
        return Loss_gradient

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B):
        Loss_intensity = F.l1_loss(image_A, image_B)
        return Loss_intensity
    
class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()      

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_in,loss_grad,loss_total
    