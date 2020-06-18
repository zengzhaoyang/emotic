from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import zipfile
from io import BytesIO
import time
import torch

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
    def __init__(self, folder, annname, transforms):
        super(ImageNet, self).__init__()
        f = open(folder + '/' + annname)
        self.z = ZipReader()
        self.folder = folder
        self.data = []
        for line in f:
            tmp = line.strip().split()
            label = int(tmp[1])
            namelist = self.z.namelist(folder + '/' + tmp[0])
            for name in namelist:
                if name.endswith('.JPEG'):
                    self.data.append((tmp[0], name, label))

        self.transforms = transforms

        print(self.data[:10])

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
