import matplotlib.pyplot as plt
import numpy as np
import argparse
from models import Model

def parse_args():
    parser = argparse.ArgumentParser(description='plot.py')
    parser.add_argument('--load_model', type=str, default=None, help='Name of file to load model from')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)

    model = Model(args)

    # assumes main.py was run with --eval_epochs=5
    val_x = [5 * i for i in range(len(model.val_losses))]
    train_losses = np.array(model.train_losses)
    val_losses = np.array(model.val_losses)

    # Plot G loss
    train_g_loss = train_losses[:, 0]
    val_g_loss = val_losses[:, 0]
    plt.plot(train_g_loss, label='Train G')
    plt.plot(val_x, val_g_loss, label='Val G')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("G Loss over Epochs")
    plt.savefig("g_losses.png")
    plt.clf()

    # Plot D loss
    train_d_loss = train_losses[:, 1]
    val_d_loss = val_losses[:, 1]
    plt.plot(train_d_loss, label='Train D')
    plt.plot(val_x, val_d_loss, label='Val D')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("D Loss over Epochs")
    plt.savefig("d_losses.png")
    plt.clf()

if __name__ == "__main__":
    main()
