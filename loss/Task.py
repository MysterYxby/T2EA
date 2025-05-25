#lossfunction
#!/usr/bin/python
# -*- encoding: utf-8 -*-
#导入要用到的包
import torch
import torch.nn as nn
import torch.nn.functional as F

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class NormalLoss(nn.Module):
    def __init__(self,ignore_lb=255, *args, **kwargs):
        super( NormalLoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        return torch.mean(loss)

class Sobelxy_g(nn.Module):
    def __init__(self):
        super(Sobelxy_g, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
    

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy_g()

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