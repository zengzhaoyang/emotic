from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Emotic(Dataset):
    def __init__(self, path, transforms):
        super(Emotic, self).__init__()
        f = open(path)
        self.data = []
        for line in f:
            tmp = line.strip().split()
            name = tmp[0]
            xmin = int(float(tmp[1]))
            ymin = int(float(tmp[2]))
            xmax = int(float(tmp[3]))
            ymax = int(float(tmp[4]))
            bbox = [xmin, ymin, xmax, ymax]
            labels = tmp[5:]
            labels = [int(item) for item in labels]
            labels = np.array(labels, dtype=np.float32)
            self.data.append((name, bbox, labels))

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        name = d[0]
        bbox = d[1]
        label = d[2]
        img = Image.open('data/emotic/emotic/' + name).convert('RGB')
        img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        img = self.transforms(img)
        return img, label
