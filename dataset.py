from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import zipfile
from io import BytesIO
import time
import torch

from skimage.color import rgb2lab

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ZipReader(object):
    def __init__(self, fname):
        self.id_context = dict()
        self.fname = fname
    def read(self, image_name):
        if self.fname in self.id_context:
            return self.id_context[self.fname].read(image_name)
        else:
            file_handle = zipfile.ZipFile(self.fname, 'r')
            self.id_context[self.fname] = file_handle
            return self.id_context[self.fname].read(image_name)
           

class ImageNet(Dataset):
    def __init__(self, zipname, annname, transforms):
        super(ImageNet, self).__init__()
        f = open(annname)
        self.data = []
        for line in f:
            tmp = line.strip().split()
            self.data.append((tmp[0], int(tmp[1])))

        self.transforms = transforms
        #self.z = zipfile.ZipFile(fname, 'r')
        self.z = ZipReader(zipname)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        name = d[0]
        label = d[1]
        bytes_img = self.z.read(name)
        img = Image.open(BytesIO(bytes_img)).convert('RGB')

        img = self.transforms(img)
        return img, label
