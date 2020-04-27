import argparse
from dataset import ImageDataset
import numpy as np
import random
import sys
import torch
from torch.utils import data

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Name of file to load model from')
    parser.add_argument(
        '--eval_epochs',
        type=int,
        default=5,
        help='How often (in epochs) to evaluate model')
    parser.add_argument(
        '--save_epochs',
        type=int,
        default=5,
        help='How often (in epochs) to save model')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Name of directory to get image data')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='saved_models',
        help='Name of directory to save models')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='RNG seed')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs to train for')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    sys.stdout.flush()

    random.seed(args.seed)
    np.random.seed(args.seed)

    full_dataset = ImageDataset(args.data_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size   = int(0.1 * len(full_dataset))
    test_size  = len(full_dataset) - train_size - val_size
    train_ds, val_ds, test_ds = data.random_split(full_dataset, [train_size, val_size, test_size])

    train_dl = data.DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

    """
    model = Model(args, train_dataset, test_dataset)
    model.train()
    """

if __name__ == "__main__":
    main()
