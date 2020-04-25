import argparse
import numpy as np
import pickle
import random
import sys
import time
import torch
import torch.nn as nn

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

    # TODO: datasets

    model = Model(args, train_dataset, test_dataset)
    model.train()

if __name__ == "__main__":
    m
