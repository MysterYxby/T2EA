#此处编写常用函数代码:数据读取，数据类型转化，图像保存等
# -*- coding: utf-8 -*-
# @Time    : 2023/5/30 16:56
# @Author  : MysterY

# 数据处理
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
from matplotlib import pyplot as plt

class MyDataset(Dataset):  # 继承了Dataset子类
    def __init__(self, ir_root, vis_root):
        # 分别读取输入/标签图片的路径信息
        self.ir_root = ir_root
        self.ir_files = os.listdir(self.ir_root)  # 列出指定路径下的所有文件

        self.vis_root = vis_root
        self.vis_files = os.listdir(self.vis_root)

        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize([256,256])])  # 将图片转换为Tensor,归一化至[0,1]
        # self.transforms = transforms.Compose([transforms.ToTensor()])  # 将图片转换为Tensor,归一化至[0,1]

    def __len__(self):
        # 获取数据集大小(两个数据集大小相同，只需回传一个)
        return len(self.ir_files)

    def __getitem__(self, index):
        # 根据索引(id)读取对应的图片
        ir_img_path = os.path.join(self.ir_root, self.ir_files[index])
        ir_img = Image.open(ir_img_path)

        vis_img_path = os.path.join(self.vis_root, self.vis_files[index])
        vis_img = Image.open(vis_img_path).convert('L')

        
        # transforms理，然后再返回最后结果
        ir_img = self.transforms(ir_img)
        vis_img = self.transforms(vis_img)

        return (ir_img, vis_img)  # 返回成对的数据

def tensor_rgb2ycbcr(img_rgb):
	R = img_rgb[:,0, :, :]
	G = img_rgb[:,1, :, :]
	B = img_rgb[:,2, :, :]
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
	return Y.unsqueeze(0), Cb.unsqueeze(0), Cr.unsqueeze(0)

def tensor_ycbcr2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128 / 255.0)
    G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    B = Y + 1.772 * (Cb - 128 / 255.0)
    # R = R.unsqueeze(1)
    # G = G.unsqueeze(1)
    # B = B.unsqueeze(1)
    # R = np.expand_dims(R, axis=-1)
    # G = np.expand_dims(G, axis=-1)
    # B = np.expand_dims(B, axis=-1)
    return torch.cat([R,G,B],1)

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def normalize1 (img):
   
    img = torch.where(img > 255, 255,img)
    img = torch.where(img < 0, 0,img)
    minv = img.min()
    maxv = img.max()
    return (img-minv)/(maxv-minv)
    # return img 

def toPIL(img):
    img = normalize1(img)
    TOPIL = transforms.ToPILImage()
    img = img.squeeze()
    return TOPIL(img)


def save_PIL(img,path):
    img  = toPIL(img)
    img.save(path)

def transform_img(img):
    trans = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Resize([256,256])
    ])
    return trans(img).unsqueeze(0)

def save_seg(seg_map,save_path):

    # 创建一个调色板（palette）来存储颜色，根据需要自定义颜色
    palette = [
        0, 0, 0,        # 背景，黑色
        255, 0, 0,      # Car，红色
        0, 255, 0,      # Person，绿色
        0, 0, 255,      # Bike，蓝色
        255, 255, 0,    # Curve，黄色
        255, 0, 255,    # car stop，品红
        0, 255, 255,    # guardrail，青色
        128, 128, 128,  # color cone，灰色
        255, 255, 255   # bump，白色
    ]

    # 获取分割结果张量的形状
    batch_size, num_classes, height, width = seg_map.shape

    # 获取分割结果张量中最大值所在的索引，即预测的类别
    _, seg_indexes = seg_map.max(dim=1)

    # 将预测的类别作为图像的像素值，并将其转换为numpy数组
    seg_indexes = seg_indexes.squeeze().byte().cpu().numpy()

    # 创建一个带有调色板的彩色图像
    seg_image = Image.fromarray(seg_indexes, mode="P")

    # 设置调色板
    seg_image.putpalette(palette)

    # 将彩色图像转换为位深度为8的RGB图像
    seg_image = seg_image.convert("RGB")


    # 保存图像到本地计算机
    seg_image.save(save_path)

def Show_Feature(feature_map):
    # 1 将传入的特征图给到f1，os:单纯为了好记，可以直接用feature_map
    f1 = feature_map

    # 2 确认特征图的shape.[B,H,W,C]
    print(f1.shape)

    # 3 预期希望的特征图shape [B,C,H,W]
    #   明显特征图shape是[B,H,W,C],利用permute进行调整
    f1 = f1.transpose(0, 3, 1, 2)

    # 4 确认特征图的shape [B,C,H,W]
    print(f1.shape)

    # 5 特征图向量从cuda转换到cpu，numpy格式
    #   自行检查特征向量位置，亦可根据报错进行修改
    #   目的 torch.Size([B,C,H,W]) 转换成 （B,C,H,W）
    #   可尝试  f1.cpu().numpy()
    # f1 = f1.cpu().detach().numpy()

    # 6 确认特征图的shape （B,C,H,W）
    print(f1.shape)

    # 7 去除B （C,H,W）
    f1 = f1.squeeze(0)

    # 8 确认特征图的shape （C,H,W）
    print(f1.shape)

    # 9 开始规范作图
    # 特征图的数量，就是维度啦，图像通常256维，超过的需要降维！
    f1_map_num = f1.shape[0]
    # 图像行显示数量
    row_num = 3
    # 绘制图像
    plt.figure()
    # 通过遍历的方式，将通道的tensor拿出
    for index in range(1, f1_map_num + 1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(f1[index - 1])
        plt.axis('off')
        plt.imsave('tuu/' + str(index) + ".png", f1[index - 1])
    plt.show()
    return 0


if __name__ == '__main__':
    # ir_root = 'train_data/ir'
    # vis_root = 'train_data/vis'
    # dataset_train = MyDataset(ir_root,vis_root)
    # train_loader = DataLoader(dataset_train,batch_size = 4,shuffle = True, num_workers = 0)
    # for batch_idx,(ir_img,vis_img) in enumerate (train_loader):
    #     ir = ir_img
    #     vis = vis_img
    #     print(batch_idx,ir.shape)
    #     print(batch_idx,vis.shape)
    for i in range(1,3):
        print(i)