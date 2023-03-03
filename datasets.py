import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

def load_img_path(dataset_path, is_train=True, model=None):
    A_path = []
    B_path = []
    L_path = []
    is_train_val = True
    is_train = is_train_val
    f1 = open(os.path.join(dataset_path, "image.txt"), 'r')
    f2 = open(os.path.join(dataset_path, "image2.txt"), 'r')
    if model == "SC":
        for line in f1.readlines():
            A_path.append(line.strip().split(" ")[0])
        f1.close()
        for line in f2.readlines():
            B_path.append(line.strip().split(" ")[0])
        f2.close()
    else:
        for line in f1.readlines():
            A_path.append(line.strip())
        f1.close()
        for line in f2.readlines():
            B_path.append(line.strip())
        f2.close()
    if is_train:
        if model == "SC":
            f1 = open(os.path.join(dataset_path, "image.txt"), 'r')
            for line in f1.readlines():
                L_path.append(int(line.strip().split(" ")[-1])) # format image_path label--> xx/xx.png 0 or xx/xx.png 1
            f1.close()
        else:
            f3 = open(os.path.join(dataset_path, "label.txt"), 'r')
            for line in f3.readlines():
                L_path.append(line.strip())
            f3.close()
    return A_path, B_path, L_path

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, transforms_L=None, is_train=True, model=None):
        self.transform = transforms.Compose(transforms_)
        self.transforml = transforms.Compose(transforms_L)
        self.transform_l = self.transform
        self.model = model
        # self.files = sorted(glob.glob(root + '/*.*'))
        self.list_A, self.list_B, self.list_L = load_img_path(dataset_path=root, is_train=is_train, model=model)
        # print("---------------------------------")
        # print(len(self.list_A))
        # print(len(self.list_B))
        # print(len(self.list_L))
        # print("---------------------------------")

    def __getitem__(self, index):
        # name = int(self.files[index].split('/')[-1].split('.')[0])
        # img = Image.open(self.files[index]).convert('RGB')

        name = os.path.split(self.list_A[index])[-1]
        img1 = Image.open(self.list_A[index]).convert('RGB')
        # print(index)
        img2 = Image.open(self.list_B[index]).convert('RGB')
        # Image.fromarray(np.uint8()).save(os.path.join("./AugMix", uname[0].split(".")[0]+"_d1.jpg"))
        # img2.save(os.path.join("./AugMix", "%d_d1.jpg" % index))
        if self.model == "SC":
            # self.transform_l = transforms.ToTensor()
            # self.transform_l = transforms.Compose([transforms.ToTensor()])
            labl = self.list_L[index]
            return self.transform(img1), self.transform(img2),np.array(img1), np.array(img2), labl, name
        else:
            labl = Image.open(self.list_L[index]).convert('P')
        return self.transform(img1), self.transform(img2), np.array(img1), np.array(img2), self.transforml(labl), name

    def __len__(self):
        return len(self.list_L)  # len(self.files1)

class ImageDataset_test(Dataset):
    def __init__(self, root, transforms_=None, transforms_L=None, is_train=False, model=None):
        self.transform = transforms.Compose(transforms_)
        self.transforml = transforms.Compose(transforms_L)
        self.model = model
        self.list_A, self.list_B, self.list_L = load_img_path(dataset_path=root, is_train=is_train, model=model)

    def __getitem__(self, index):
        name = os.path.split(self.list_A[index])[-1]
        img1 = Image.open(self.list_A[index]).convert('RGB')
        img2 = Image.open(self.list_B[index]).convert('RGB')
        # labl = Image.open(self.list_L[index]).convert('P')
        if self.model == "SC":
            labl = self.list_L[index]
            return self.transform(img1), self.transform(img2), labl, name
        else:
            labl = Image.open(self.list_L[index]).convert('P')
        return self.transform(img1), self.transform(img2), self.transforml(labl), name
    def __len__(self):
        return len(self.list_L)  # len(self.files1)

# Configure dataloaders

def Get_dataloader(path,batch, reshape_size, model=None):
    #Image.BICUBIC
    transforms_ = [transforms.Resize(reshape_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

    transforms_L = [transforms.Resize(reshape_size),
                transforms.ToTensor(),
                ]

    train_dataloader = DataLoader(
        ImageDataset(path, transforms_=transforms_, transforms_L=transforms_L, model=model),
        batch_size=batch, shuffle=True, num_workers=2, drop_last=True)
    return train_dataloader

def Get_dataloader_test(path,batch, reshape_size=None, model=None):
    transforms_ = [transforms.Resize(reshape_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    transforms_L = [transforms.Resize(reshape_size),
                transforms.ToTensor(),
                ]

    test_dataloader = DataLoader(
        ImageDataset_test(path, transforms_=transforms_, transforms_L=transforms_L, model=model),
        batch_size=batch, shuffle=False, num_workers=2, drop_last=False)

    return test_dataloader


