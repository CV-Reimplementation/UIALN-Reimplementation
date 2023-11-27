# DataLoader
import cv2
import numpy as np
import torch.random
from torch.utils.data import Dataset
from torchvision import transforms
import os


def get_transform_0(size=None):
    if size is not None:
        transform = transforms.Compose([
            # RGB转化为LAB
            transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
            # 只保留L通道
            transforms.Lambda(lambda x: x[:, :, 0]),
            transforms.ToTensor(),
            transforms.Resize((size, size))
        ])
    else:
        transform = transforms.Compose([
            # RGB转化为LAB
            transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
            # 只保留L通道
            transforms.Lambda(lambda x: x[:, :, 0]),
            transforms.ToTensor(),
        ])
    return transform

def get_transform_1(size=None):
    if size is not None:
        transform = transforms.Compose([
            # RGB转化为LAB
            transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
            # 只保留AB通道
            transforms.Lambda(lambda x: x[:, :, 1:]),
            transforms.ToTensor(),
            transforms.Resize((size, size))
        ])
    else:
        transform = transforms.Compose([
            # RGB转化为LAB
            transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
            # 只保留AB通道
            transforms.Lambda(lambda x: x[:, :, 1:]),
            transforms.ToTensor(),
        ])
    return transform

def get_transform_lab(size=None):
    if size is not None:
        transform = transforms.Compose([
            # RGB转化为LAB
            transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
            transforms.ToTensor(),
            transforms.Resize((size, size))
        ])
    else:
        transform = transforms.Compose([
            # RGB转化为LAB
            transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
            transforms.ToTensor(),
        ])
    return transform


# 分解数据集，以及I_delight部分的数据集
class retinex_decomposition_data(Dataset):
    def __init__(self, I_no_light_path, I_light_path):
        self.I_light_imglist = self.get_path(I_light_path)
        self.I_no_light_imglist = [os.path.join(I_no_light_path, os.path.basename(img_path)) for img_path in
                                       self.I_light_imglist]

        self.transform = get_transform_0()

    def get_path(self, path):
        img_name_list = sorted(os.listdir(path))
        img_list = []
        for img_name in img_name_list:
            img_list.append(os.path.join(path, img_name))
        return img_list

    def __len__(self):
        return len(self.I_no_light_imglist)

    def __getitem__(self, index):
        I_no_AL_img_path = self.I_no_light_imglist[index]
        I_AL_img_path = self.I_light_imglist[index]

        I_no_AL_img = cv2.imread(I_no_AL_img_path, cv2.IMREAD_COLOR)
        I_AL_img = cv2.imread(I_AL_img_path, cv2.IMREAD_COLOR)

        # 检查图片是否读取成功
        if I_no_AL_img is None or I_AL_img is None:
            print(index)
            print(I_AL_img_path)
            print(I_AL_img)
            print("Error: 图片读取失败")
            exit(0)

        I_no_AL_img = cv2.cvtColor(I_no_AL_img, cv2.COLOR_BGR2RGB)
        I_AL_img = cv2.cvtColor(I_AL_img, cv2.COLOR_BGR2RGB)

        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        I_no_AL_tensor = self.transform(I_no_AL_img)
        torch.random.manual_seed(seed)
        I_AL_tensor = self.transform(I_AL_img)

        return I_no_AL_tensor, I_AL_tensor

# AL区域自导向色彩恢复模块数据集
class AL_data(Dataset):
    def __init__(self, ABcc_path, gt_path, size=256):
        self.size = size
        self.ABcc_imglist = self.get_path(ABcc_path)
        # gt_name是basename的_前面的部分
        # self.gt_imglist = [os.path.join(gt_path, os.path.basename(img_path).split("_")[0]+'.bmp') for img_path in self.ABcc_imglist]
        self.gt_imglist = [os.path.join(gt_path, os.path.basename(img_path)) for img_path in
                           self.ABcc_imglist]

        self.transform_1 = get_transform_1(self.size)
        self.transform_0 = get_transform_0(self.size)

    def get_path(self, path):
        img_name_list = sorted(os.listdir(path))
        img_list = []
        for img_name in img_name_list:
            img_list.append(os.path.join(path, img_name))
        return img_list

    def __len__(self):
        return len(self.ABcc_imglist)

    def __getitem__(self, index):
        ABcc_img_path = self.ABcc_imglist[index]
        gt_img_path = self.gt_imglist[index]

        ABcc_img = cv2.imread(ABcc_img_path, cv2.IMREAD_COLOR)
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)

        # 检查图片是否读取成功
        if ABcc_img is None or gt_img is None:
            print(index)
            print(ABcc_img_path)
            print(gt_img_path)
            print("Error: 图片读取失败")
            exit(0)

        ABcc_img = cv2.cvtColor(ABcc_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        ABcc_tensor = self.transform_1(ABcc_img)
        torch.random.manual_seed(seed)
        gt_tensor = self.transform_1(gt_img)
        torch.random.manual_seed(seed)
        L_tensor = self.transform_0(ABcc_img)

        return ABcc_tensor, gt_tensor, L_tensor

class Detail_Enhancement_data(Dataset):
    def __init__(self, ABcc_path, gt_path, size=256):
        self.size = size
        self.ABcc_imglist = self.get_path(ABcc_path)
        # gt_name是basename的_前面的部分
        # self.gt_imglist = [os.path.join(gt_path, os.path.basename(img_path).split("_")[0]+'.bmp') for img_path in self.ABcc_imglist]
        self.gt_imglist = [os.path.join(gt_path, os.path.basename(img_path)) for img_path in self.ABcc_imglist]
        self.transform_1 = get_transform_1(self.size)
        self.transform_0 = get_transform_0(self.size)
        self.transform_lab = get_transform_lab(self.size)

    def get_path(self, path):
        img_name_list = sorted(os.listdir(path))
        img_list = []
        for img_name in img_name_list:
            img_list.append(os.path.join(path, img_name))
        return img_list

    def __len__(self):
        return len(self.ABcc_imglist)

    def __getitem__(self, index):
        ABcc_img_path = self.ABcc_imglist[index]
        gt_img_path = self.gt_imglist[index]

        ABcc_img = cv2.imread(ABcc_img_path, cv2.IMREAD_COLOR)
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)

        # 检查图片是否读取成功
        if ABcc_img is None or gt_img is None:
            print(index)
            print(ABcc_img_path)
            print(gt_img_path)
            print("Error: 图片读取失败")
            exit(0)

        ABcc_img = cv2.cvtColor(ABcc_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        ABcc_tensor = self.transform_1(ABcc_img)
        torch.random.manual_seed(seed)
        gt_L_tensor = self.transform_0(gt_img)
        torch.random.manual_seed(seed)
        L_tensor = self.transform_0(ABcc_img)
        torch.random.manual_seed(seed)
        gt = self.transform_lab(gt_img)

        return ABcc_tensor, L_tensor, gt_L_tensor, gt


class Test_data(Dataset):
    def __init__(self, input_path, size=256):
        self.size = size
        self.input_imglist = self.get_path(input_path)

        self.transform_1 = get_transform_1(self.size)
        self.transform_0 = get_transform_0(self.size)
        self.transform_lab = get_transform_lab(self.size)

    def get_path(self, path):
        img_name_list = sorted(os.listdir(path))
        img_list = []
        for img_name in img_name_list:
            img_list.append(os.path.join(path, img_name))
        return img_list

    def __len__(self):
        return len(self.input_imglist)

    def __getitem__(self, index):
        input_img_path = self.input_imglist[index]
        img_name = os.path.basename(input_img_path)

        input_img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)

        # 检查图片是否读取成功
        if input_img is None:
            print(index)
            print(input_img_path)
            print("Error: 图片读取失败")
            exit(0)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        AB_tensor = self.transform_1(input_img)

        torch.random.manual_seed(seed)
        L_tensor = self.transform_0(input_img)

        return AB_tensor, L_tensor, img_name


if __name__ == "__main__":
    '''
    # 使用retinex_decomposition_data(Dataset)随机取出一组查看  
    I_no_light_path = r"./dataset/UIALN_dataset/train_data/dataset_no_AL"
    I_light_path = r"./dataset/UIALN_dataset/train_data/dataset_with_AL/train"
    dataset = retinex_decomposition_data(I_no_light_path, I_light_path)
    I_no_AL_tensor, I_AL_tensor = dataset[50]
    # 转化为image
    I_no_AL_img = transforms.ToPILImage()(I_no_AL_tensor)
    I_AL_img = transforms.ToPILImage()(I_AL_tensor)
    I_no_AL_img.show()
    I_AL_img.show()'''

    # 使用AL_data(Dataset)随机取出一组查看
    ABcc_path = r"dataset/UIALN_dataset/train_data/dataset_with_AL/train"
    gt_path = r"dataset/UIALN_dataset/train_data/labels/raw"
    dataset = AL_data(ABcc_path, gt_path)
    ABcc_tensor, gt_tensor = dataset[90]
    # 转化为image
    ABcc_img = transforms.ToPILImage()(ABcc_tensor)
    gt_img = transforms.ToPILImage()(gt_tensor)
    ABcc_img.show()
    gt_img.show()
