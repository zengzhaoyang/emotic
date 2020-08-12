from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import zipfile
from io import BytesIO
import time
import torch
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ZipReader(object):
    def __init__(self):
        self.id_context = dict()
    def read(self, zip_name, image_name):
        if zip_name in self.id_context:
            return self.id_context[zip_name].read(image_name)
        else:
            file_handle = zipfile.ZipFile(zip_name, 'r')
            self.id_context[zip_name] = file_handle
            return self.id_context[zip_name].read(image_name)

    def namelist(self, zip_name):
        if zip_name in self.id_context:
            return self.id_context[zip_name].namelist()
        else:
            file_handle = zipfile.ZipFile(zip_name, 'r')
            self.id_context[zip_name] = file_handle
            return self.id_context[zip_name].namelist()
         

class ImageNet(Dataset):
    def __init__(self, folder, imglist, zipfile, transforms):
        super(ImageNet, self).__init__()

        mapp = {}
        folders = set()
        f = open(folder + '/' + imglist)
        for line in f:
            tmp = line.strip().split()
            label = int(tmp[1])
            path = tmp[0].split('/')[-1]
            tmp_folder = zipfile 
            folders.add(tmp_folder)
            mapp[path] = label

        self.z = ZipReader()
        self.folder = folder
        self.data = []

        for tmp in folders:
            if os.path.exists(folder + '/' + tmp):
                namelist = self.z.namelist(folder + '/' + tmp)
                for name in namelist:
                    fname = name.split('/')[-1]
                    if name.endswith('.JPEG') and fname in mapp:
                        self.data.append((tmp, name, mapp[fname]))

        f.close()

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        zip_name = d[0]
        image_name = d[1]
        label = d[2]
        bytes_img = self.z.read(self.folder + '/' + zip_name, image_name)
        img = Image.open(BytesIO(bytes_img)).convert('RGB')

        img = self.transforms(img)
        return img, label
