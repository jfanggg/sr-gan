import argparse
from dataset import ImageDataset
from models import Model
import numpy as np
import os
import random
import sys
import torch
from torch.utils import data

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--model', type=str, default=None, help='Name of file to load model from')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--eval_epochs', type=int, default=5, help='How often (in epochs) to evaluate model')
    parser.add_argument('--save_epochs', type=int, default=5, help='How often (in epochs) to save model')
    parser.add_argument('--data_dir', type=str, default='data', help='Name of directory to get image data')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Name of directory to save models')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    sys.stdout.flush()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataloaders = {}
    for key in ['train', 'val' , 'test']:
        dataset = ImageDataset(os.path.join(args.data_dir, key))
        dataloaders[key] = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = Model(args)
    model.train(dataloaders)

if __name__ == "__main__":
    main()
