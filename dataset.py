from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import zipfile
from io import BytesIO
import time

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
           

class Emotic(Dataset):
    def __init__(self, fname, path, transforms):
        super(Emotic, self).__init__()
        f = open(path)
        self.data = []
        for line in f:
            tmp = line.strip().split()
            if 'train' in path and tmp[-1] == 'nan':
                continue
            name = tmp[0]
            xmin = int(float(tmp[1]))
            ymin = int(float(tmp[2]))
            xmax = int(float(tmp[3]))
            ymax = int(float(tmp[4]))
            bbox = [xmin, ymin, xmax, ymax]
            labels = tmp[5:]
            labels = [float(item) for item in labels]
            labels = np.array(labels, dtype=np.float32)
            self.data.append((name, bbox, labels))

        self.transforms = transforms

        #self.z = zipfile.ZipFile(fname, 'r')
        self.z = ZipReader(fname)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        name = d[0]
        bbox = d[1]
        label = d[2]
        bytes_img = self.z.read('emotic/' + name)
        #img = Image.open('data/emotic/emotic/' + name).convert('RGB')
        img = Image.open(BytesIO(bytes_img)).convert('RGB')
        img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        img = self.transforms(img)
        return img, label
