# 数据处理
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import cv2 
import glob

class MyDataset(Dataset):  # 继承了Dataset子类
    def __init__(self, ir_root, vis_root):
        # 分别读取输入/标签图片的路径信息
        self.ir_root = ir_root
        self.ir_files = os.listdir(self.ir_root)  # 列出指定路径下的所有文件

        self.vis_root = vis_root
        self.vis_files = os.listdir(self.vis_root)

        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Grayscale(),
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
    
class MyDataset_OneImg(Dataset):  # 继承了Dataset子类
    def __init__(self, root):
        # 分别读取输入/标签图片的路径信息
        self.root = root
        self.files = os.listdir(self.root)  # 列出指定路径下的所有文件


        self.transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Grayscale(),
                                              transforms.Resize([256,256])])  # 将图片转换为Tensor,归一化至[0,1]
      

    def __len__(self):
        # 获取数据集大小(两个数据集大小相同，只需回传一个)
        return len(self.files)

    def __getitem__(self, index):
        # 根据索引(id)读取对应的图片
        img_path = os.path.join(self.root, self.files[index])
        img = Image.open(img_path)


        
        # transforms理，然后再返回最后结果
        img = self.transforms(img)
    
        return (img)  # 返回成对的数据
    

#Dataloader
def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            data_dir_vis = 'code/data/Seg/vis/'
            data_dir_ir = 'code/data/Seg/inf/'
            data_dir_label = 'code/data/Seg/label/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            label = np.array(Image.open(label_path))
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            label = np.asarray(Image.fromarray(label), dtype=np.int64)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(label),
                name,
            )
        elif self.split=='val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )

    def __len__(self):
        return self.length