import math
import numpy as np
import cv2

#model parameters 模型参数
L = 1;
Tg = 0.9994;
kg = -15;
Dg = 0.5;
Ta = 0.9879;
ka = -22;
Da = 0.8;

#Sobel Operator Sobel算子
h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

#if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
#如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

# strA = cv2.imread("images/MDDataset/c01_1.tif", 0).astype(np.float32)
# strB = cv2.imread("images/MDDataset/c01_2.tif", 0).astype(np.float32)
# strF = cv2.imread("results/our/guided_1.png", 0).astype(np.float32)


# 数组旋转180度
def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

#相当于matlab的Conv2
def convolution(k, data):
    k = flip180(k)
    data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    n,m = data.shape
    img_new = []
    for i in range(n-2):
        line = []
        for j in range(m-2):
            a = data[i:i+3,j:j+3]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)
    return np.array(img_new)


#用h3对strA做卷积并保留原形状得到SAx，再用h1对strA做卷积并保留原形状得到SAy
#matlab会对图像进行补0，然后卷积核选择180度
#gA = sqrt(SAx.^2 + SAy.^2);
#定义一个和SAx大小一致的矩阵并填充0定义为aA，并计算aA的值
def getArray(img):
    SAx = convolution(h3,img)
    SAy = convolution(h1,img)
    gA = np.sqrt(np.multiply(SAx,SAx)+np.multiply(SAy,SAy))
    n, m = img.shape
    aA = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if(SAx[i,j]==0):
                aA[i,j] = math.pi/2
            else:
                aA[i, j] = math.atan(SAy[i,j]/SAx[i,j])
    return gA,aA



#the relative strength and orientation value of GAF,GBF and AAF,ABF;
def getQabf(aA,gA,aF,gF):
    n, m = aA.shape
    GAF = np.zeros((n,m))
    AAF = np.zeros((n,m))
    QgAF = np.zeros((n,m))
    QaAF = np.zeros((n,m))
    QAF = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if(gA[i,j]>gF[i,j]):
                GAF[i,j] = gF[i,j]/gA[i,j]
            elif(gA[i,j]==gF[i,j]):
                GAF[i, j] = gF[i, j]
            else:
                GAF[i, j] = gA[i,j]/gF[i, j]
            AAF[i,j] = 1-np.abs(aA[i,j]-aF[i,j])/(math.pi/2)

            QgAF[i,j] = Tg/(1+math.exp(kg*(GAF[i,j]-Dg)))
            QaAF[i,j] = Ta/(1+math.exp(ka*(AAF[i,j]-Da)))

            QAF[i,j] = QgAF[i,j]*QaAF[i,j]

    return QAF


def get_Qabf(strA,strB,strF):
    #对strB和strF进行相同的操作
    gA,aA = getArray(strA)
    gB,aB = getArray(strB)
    gF,aF = getArray(strF)
    QAF = getQabf(aA,gA,aF,gF)
    QBF = getQabf(aB,gB,aF,gF)

    #计算QABF
    deno = np.sum(gA+gB)
    nume = np.sum(np.multiply(QAF,gA)+np.multiply(QBF,gB))

    output = nume/deno
    return output