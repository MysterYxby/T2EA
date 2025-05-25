from torch.utils.data import Dataset
from torchvision import transforms
import torch

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
        #transforms.Resize([256,256])
    ])
    return trans(img).unsqueeze(0)
