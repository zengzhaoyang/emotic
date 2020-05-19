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
           
class ToLAB(object):

    def __init__(self):
        pass

    def __call__(self, img):
        rgb_image = np.array(img)
        lab_image = rgb2lab(rgb_image)
        l_image = (np.clip(lab_image[:, :, 0:1], 0.0, 100.0) + 0.0) / (100.0 + 0.0)
        a_image = (np.clip(lab_image[:, :, 1:2], -86.0, 98.0) + 86.0) / (98.0  + 86.0)
        b_image = (np.clip(lab_image[:, :, 2:3], -107.0, 94.0) + 107.0) / (94.0 + 107.0)
        img = np.concatenate((l_image, a_image, b_image), axis=2).astype(np.float32)
        tensor = torch.from_numpy(img)
        tensor = tensor.permute(2, 0, 1)
        return tensor


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

        #rgb_image = np.array(img)
        #lab_image = rgb2lab(rgb_image)
        #l_image = (np.clip(lab_image[:, :, 0:1], 0.0, 100.0) + 0.0) / (100.0 + 0.0)
        #a_image = (np.clip(lab_image[:, :, 1:2], -86.0, 98.0) + 86.0) / (98.0  + 86.0)
        #b_image = (np.clip(lab_image[:, :, 2:3], -107.0, 94.0) + 107.0) / (94.0 + 107.0)
        #img = np.concatenate((l_image, a_image, b_image), axis=2).astype(np.float32)

        img = self.transforms(img)
        return img, label
