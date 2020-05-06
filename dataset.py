from constants import IMAGE_SIZE
import glob
import numpy as np
import os
from PIL import Image
import torch
from torch.utils import data

class ImageDataset(data.Dataset):
    def __init__(self, path):
        self.files = [f for f in glob.glob(os.path.join(path, "*.png"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx])

        small_size = int(IMAGE_SIZE / 4)
        high_res = np.array(im).astype(np.float64)
        low_res  = np.array(im.resize((small_size, small_size), Image.BICUBIC)).astype(np.float64)

        # swap [H, W, C] to [C, H, W]
        low_res  = low_res.transpose((2, 0, 1))
        high_res = high_res.transpose((2, 0, 1))

        # convert low res to [0, 1] and high_res to [-1, 1]
        low_res = low_res / 255.0
        high_res = 2 * high_res / 255.0 - 1
        
        return {'low_res' : torch.from_numpy(low_res).float(),
                'high_res': torch.from_numpy(high_res).float()}
