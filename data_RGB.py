import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
from glob import glob

ImageFile.LOAD_TRUNCATED_IMAGES=True
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir='./', patch_size=256):
        super(DataLoaderTrain, self).__init__()

        self.inp_filenames = sorted(glob(rgb_dir + 'train/source/*_*'))
        self.tar_filenames = sorted(glob(rgb_dir + 'train/target/*_*'))
        assert len(self.inp_filenames) == len(self.tar_filenames)
        self.sizex = len(self.tar_filenames)
        self.ps = patch_size

    def __len__(self):
        return self.sizex
    
    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps
        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        
        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')
        tep_inp_img = TF.to_tensor(inp_img)
        tep_tar_img = TF.to_tensor(tar_img)

        hh, ww = tep_tar_img.shape[1], tep_tar_img.shape[2]
        
        
        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)
        
        
        inp_img = tep_inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tep_tar_img[:, rr:rr+ps, cc:cc+ps]
        
        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))

        label = index_ // (self.sizex // 4)

        return inp_img, tar_img, label, index_


class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir='./', patch_size = None):
        super(DataLoaderTest, self).__init__()
        self.inp_filenames = sorted(glob(rgb_dir + 'test/source/*_*'))
        self.tar_filenames = sorted(glob(rgb_dir + 'test/target/*_*'))
        assert len(self.inp_filenames) == len(self.tar_filenames)
        self.sizex       = len(self.tar_filenames)
        #self.ps = 256
        self.ps = patch_size

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        #print(inp_img.size)
        # Validate on center crop
        if self.ps is not None and ps <= inp_img.size[0] and ps <= inp_img.size[1]:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]
        label = index_ // (self.sizex // 4)
        return inp_img, tar_img, label, filename
