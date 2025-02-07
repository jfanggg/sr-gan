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
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'generate'], help='What to do with the model')

    # Training Args
    parser.add_argument('--load_model', type=str, default=None, help='Name of file to load model from')
    parser.add_argument('--pretrain_epochs', type=int, default=5, help='Number of epochs to pretrain generator for')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train for')
    parser.add_argument('--eval_epochs', type=int, default=5, help='How often (in epochs) to evaluate model')
    parser.add_argument('--save_epochs', type=int, default=5, help='How often (in epochs) to save model')
    parser.add_argument('--data_dir', type=str, default='data', help='Name of directory to get image data')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Name of directory to save models')

    # Generate args
    parser.add_argument('--generate_dir', type=str, default='generated', help='Name of directory to save generated images')

    args = parser.parse_args()
    return args

def get_dataloader(data_dir, subdir, train=None):
    full_directory = os.path.join(data_dir, subdir)
    if os.path.exists(full_directory):
        dataset = ImageDataset(full_directory, train)
        return data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
        # return data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    return None

def main():
    args = parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataloaders = {}
    dataloaders['train'] = get_dataloader(args.data_dir, 'train', train=True)
    dataloaders['val']   = get_dataloader(args.data_dir, 'val')
    dataloaders['test']  = get_dataloader(args.data_dir, 'test')

    model = Model(args)

    if args.mode == 'train':
        model.train(dataloaders['train'], dataloaders['val'])
    elif args.mode == 'evaluate':
        g_loss, d_loss = model.evaluate(dataloaders['test'])
        print("Test G loss: {:.4f} | Test D loss: {:.4f}".format(g_loss, d_loss))
    elif args.mode == 'generate':
        model.generate(dataloaders['test'])

if __name__ == "__main__":
    main()
