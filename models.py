import math
import numpy as np
import os
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model():
    def __init__(self, args):
        self.args = args

        if args.model:
            self.load_state(args.model)
        else:
            self.epoch = 0
            self.G = Generator()
            self.D = Discriminator()
            self.g_optimizer = optim.Adadelta(self.G.parameters())
            self.d_optimizer = optim.Adadelta(self.D.parameters())

        # extract all layers prior to the last softmax of VGG-19
        vgg19_layers = list(models.vgg19(pretrained = True).features)[:30]
        self.vgg19 = nn.Sequential(*vgg19_layers)

        self.content_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCELoss()

    def load_state(self, fname):
        with open(fname, 'rb') as f:
            state = pickle.load(f)
        
        self.epoch       = state["epoch"]
        self.G           = state["G"]
        self.D           = state["D"]
        self.g_optimizer = state["g_optimizer"]
        self.d_optimizer = state["d_optimizer"]

    def save_state(self):
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        fname = "%s/save" % self.args.save_dir
        state = {
            "epoch"       : self.epoch,
            "G"           : self.G,
            "D"           : self.D,
            "g_optimizer" : self.g_optimizer,
            "d_optimizer" : self.d_optimizer,
        }
        with open("%s_%d.pkl" % (fname, self.epoch), 'wb') as f:
            pickle.dump(state, f)

    def train(self, dataloaders):
        self.D.to(device)
        self.G.to(device)
        self.vgg19.to(device)

        while self.epoch < self.args.epochs:
            print("=== Epoch: %d ===" % self.epoch)
            if self.epoch % self.args.save_epochs == 0:
                self.save_state()
            if self.epoch % self.args.eval_epochs == 0:
                self.evaluate(dataloaders)

            self.D.train()
            self.G.train()
            g_losses, d_losses = [], []

            for batch in dataloaders['train']:
                low_res  = batch['low_res']
                high_res = batch['high_res']
                low_res.to(device)
                high_res.to(device)

                batch_size = high_res.size(0)
                real = torch.ones((batch_size, 1), requires_grad=False)
                fake = torch.zeros((batch_size, 1), requires_grad=False)
                real.to(device)
                fake.to(device)

                """ Generator training """
                self.g_optimizer.zero_grad()

                generated = self.G(low_res)

                content_loss = self.content_loss(self.vgg19(high_res), self.vgg19(generated))
                adversarial_loss = self.bce_loss(self.D(generated), real)

                g_loss = content_loss + 1E-3 * adversarial_loss
                g_losses.append(g_loss.item())
                g_loss.backward()
                self.g_optimizer.step()

                """ Discriminator training """
                self.d_optimizer.zero_grad()

                real_loss = self.bce_loss(self.D(high_res), real)
                fake_loss = self.bce_loss(self.D(generated.detach()), fake)

                d_loss = (real_loss + fake_loss) / 2
                d_losses.append(d_loss.item())
                d_loss.backward()
                self.d_optimizer.step()

            if self.epoch % self.args.eval_epochs == 0:
                print("Train G loss: ", np.mean(g_losses))
                print("Train D loss: ", np.mean(d_losses))

            sys.stdout.flush()
            self.epoch += 1

        print("Finished training")
        self.save_state()
        self.evaluate(dataloaders)
        sys.stdout.flush()

    def evaluate(self, dataloaders):
        # TODO
        pass

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        return x + self.res_block(x)

def shuffle_block():
    return nn.Sequential(
        nn.Conv2d(64, 256, 3, padding=1),
        nn.PixelShuffle(2),
        nn.PReLU()
    )

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.B = 16
        self.S = 2

        self.conv1      = nn.Sequential(nn.Conv2d(3, 64, 9, padding=4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock() for _ in range(self.B)])
        self.conv2      = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.shuffle    = nn.Sequential(*[shuffle_block() for _ in range(self.S)])
        self.conv3      = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x):
        """
        Params:
        - x ([N, 3, H, W] Tensor): batch of images to super-resolve

        Returns:
        - [N, 3, 4H, 4W] Tensor
        """

        x = self.conv1(x)
        saved_x = x
        x = self.res_blocks(x)
        x = self.conv2(x) + saved_x
        x = self.shuffle(x)
        x = self.conv3(x)
        return x

def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

def conv_size(x_in, kernel_size, stride, padding):
    return math.floor((x_in + 2 * padding - kernel_size) / stride + 1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        in_channels = 3
        im_width = 96

        layers = []
        for idx, out_channels in enumerate([64, 128, 256, 512]):
            if idx == 0:
                layers.extend([nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2)])
            else:
                layers.append(conv_block(in_channels, out_channels, 3, 1, 1))

            layers.append(conv_block(out_channels, out_channels, 3, 2, 1))

            im_width = conv_size(im_width, 3, 1, 1)
            im_width = conv_size(im_width, 3, 2, 1)
            in_channels = out_channels
        self.convs = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(im_width * im_width * 512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Params:
        - x ([N, 3, H, W] Tensor): batch of images to test
        """
        N = list(x.shape)[0]

        x = self.convs(x)
        x = torch.reshape(x, (N, -1))
        x = self.fc(x)
        return x
