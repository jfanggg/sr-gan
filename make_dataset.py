from constants import IMAGE_SIZE
import numpy as np
import os
import glob
from PIL import Image

# Takes in directory and produces random cropped images
def make_dataset(in_dir, out_dir, num_samples):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    all_files = [f for f in glob.glob(os.path.join(in_dir, "**/*.jpg"), recursive=True)]

    sampled = 0
    while sampled < num_samples:
        idx = np.random.randint(len(all_files))
        im = Image.open(all_files[idx])
        w, h = im.size

        if w <= IMAGE_SIZE or h <= IMAGE_SIZE:
            continue

        x = np.random.randint(0, w - IMAGE_SIZE + 1)
        y = np.random.randint(0, h - IMAGE_SIZE + 1)

        cropped = im.crop((x, y, x + IMAGE_SIZE, y + IMAGE_SIZE))
        cropped.save(os.path.join(out_dir, "{}.png".format(sampled)))
        sampled += 1

make_dataset("Images", "data", 50)
