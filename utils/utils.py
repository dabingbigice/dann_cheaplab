import random

import numpy as np
import torch
from PIL import Image
import numpy as np
from PIL import Image
import cv2


def cvtColor(image):
    """
    将图像转换为RGB格式
    参数:
        image: PIL Image对象
    返回:
        image: RGB格式的PIL Image对象
    """
    if len(np.array(image).shape) == 2:
        # 如果是灰度图，转换为RGB
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        # 如果是其他模式（如RGBA），转换为RGB
        image = image.convert('RGB')
    return image


def preprocess_input(x):
    """
    图像预处理函数，进行归一化处理
    参数:
        x: numpy数组，形状为(H, W, C)，值范围[0, 255]
    返回:
        x: 预处理后的numpy数组，值范围[-1, 1]
    """
    # 确保数据类型为float32
    x = np.array(x, dtype=np.float32)

    # 如果值范围是[0, 255]，归一化到[0, 1]
    if x.max() > 1.0:
        x /= 255.0

    # 进一步归一化到[-1, 1]（常用在深度学习模型中）
    # 或者使用ImageNet统计量进行标准化
    mean = [0.485, 0.456, 0.406]  # ImageNet均值
    std = [0.229, 0.224, 0.225]  # ImageNet标准差

    # 应用标准化
    x[..., 0] = (x[..., 0] - mean[0]) / std[0]  # R通道
    x[..., 1] = (x[..., 1] - mean[1]) / std[1]  # G通道
    x[..., 2] = (x[..., 2] - mean[2]) / std[2]  # B通道

    return x


def preprocess_input_simple(x):
    """
    简化的预处理函数，只进行简单的归一化
    参数:
        x: numpy数组
    返回:
        x: 归一化到[0, 1]的数组
    """
    x = np.array(x, dtype=np.float32)
    if x.max() > 1.0:
        x /= 255.0
    return x


def preprocess_input_imagenet(x):
    """
    使用ImageNet统计量进行标准化
    参数:
        x: numpy数组，形状为(H, W, C)
    返回:
        x: 标准化后的数组
    """
    x = np.array(x, dtype=np.float32)

    # 如果值范围是[0, 255]，先归一化到[0, 1]
    if x.max() > 1.0:
        x /= 255.0

    # ImageNet统计量
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 应用标准化
    x[..., 0] = (x[..., 0] - mean[0]) / std[0]  # R通道
    x[..., 1] = (x[..., 1] - mean[1]) / std[1]  # G通道
    x[..., 2] = (x[..., 2] - mean[2]) / std[2]  # B通道

    return x
#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)