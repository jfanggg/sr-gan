import numpy as np
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model():
    def __init__(self, args, train_dataset, test_dataset):
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # CUDA stuff

        if args.model:
            self.load_state(args.model)
        else:
            self.network = # TODO
            self.epoch = 0
            self.optimizer = optim.Adadelta(self.network.parameters())
        self.loss_function = nn.BCELoss()
        self.network.to(device)

    def load_state(self, fname):
        with open(fname, 'rb') as f:
            state = pickle.load(f)
        
        self.network    = state["network"]
        self.epoch      = state["epoch"]
        self.optimizer  = state["optimizer"]

    def save_state(self):
        fname = "%s/save" % self.args.save_dir
        state = {
            "epoch"     : self.epoch,
            "network"   : self.network,
            "optimizer" : self.optimizer,
        }
        with open("%s_%d.pkl" % (fname, self.epoch), 'wb') as f:
            pickle.dump(state, f)

    def train(self):
        while self.epoch < self.args.epochs:
            print("=== Epoch: %d ===" % self.epoch)

            if self.epoch % self.args.save_epochs == 0:
                self.save_state()
            if self.epoch % self.args.eval_epochs == 0:
                self.evaluate()

            losses = []
            self.network.train()
            for t in range(0, len(self.train_dataset) - SEQ_LEN, DIV_LEN):
                self.network.zero_grad()
                """
                self.network.init_hidden()
                
                # convert raw data to input features
                vs = self.train_dataset[t:t + SEQ_LEN - 1]
                xs = notes2inputs(vs)
                p = self.network.forward(xs)

                target = torch.tensor(self.train_dataset[t + 1:t + SEQ_LEN])
                target = target.view(SEQ_LEN - 1, MIDI_RANGE, -1).float().to(device)
                loss = self.loss_function(p, target)

                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                """

            if self.epoch % self.args.eval_epochs == 0:
                print("Train loss: ", np.mean(losses))

            sys.stdout.flush()
            self.epoch += 1

        print("Finished training")
        self.save_state()
        self.evaluate()
        sys.stdout.flush()

    def evaluate(self):
        self.network.eval()

        losses = []
        for t in range(0, len(self.test_dataset) - SEQ_LEN, SEQ_LEN):
            self.network.zero_grad()
            self.network.init_hidden()
            
            # padded
            # convert raw data to input features
            vs = self.test_dataset[t:t + SEQ_LEN - 1, MIDI_RANGE_L:MIDI_RANGE_R]
            xs = notes2inputs(vs)
            p = self.network.forward(xs)
            padded_p = torch.zeros(len(vs), 128)
            padded_p[:, MIDI_RANGE_L:MIDI_RANGE_R] = np.squeeze(p)
            padded_p = padded_p.to(device)

            target = torch.tensor(self.test_dataset[t + 1:t + SEQ_LEN])
            target = target.float().to(device)

            loss = self.loss_function(padded_p, target)
            losses.append(loss.item())
        
        mean_loss = np.mean(losses)
        return mean_loss

def res_block():
    return nn.Sequential(
        nn.Conv2D(64, 64, 3, padding=1),
        nn.BatchNorm2D(64),
        nn.PReLU(),
        nn.Conv2D(64, 64, 3, padding=1),
        nn.BatchNorm2D(64),
    )

def shuffle_block():
    return nn.Sequential(
        nn.Conv2D(64, 256, 3, padding=1),
        nn.PixelShuffle(2),
        nn.PReLU()
    )

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.B = #TODO
        self.U = 2

        self.conv1      = nn.Sequential(nn.Conv2D(3, 64, 9, padding=4), nn.PReLU())
        self.res_blocks = [res_block() for _ in range(self.B)]
        self.conv2      = nn.Sequential(nn.Conv2D(64, 64, 3, padding=1), nn.BatchNorm2D(64))
        self.shuffle    = nn.Sequential(*[shuffle_block() for _ in range(U)])
        self.conv3      = nn.Conv2D(64, 3, 9, padding=4)

    def forward(self, x):
        """
        Params:
        - x ([batch_size, 3, H, W] Tensor): batch of images to super-resolve

        Returns:
        - [batch_size, 3, 4*H, 4*W] Tensor
        """
        x = image.float().to(device)

        x = self.conv1(x)
        saved_x = x

        for r_b in self.res_blocks:
            x = r_b(x) + x

        x = self.conv2(x) + saved_x
        x = self.shuffle(x)
        x = self.conv3(x)

        return x
