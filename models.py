import numpy as np
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model():
    def __init__(self, args, train_dataset, test_dataset):
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        if args.model:
            self.load_state(args.model)
        else:
            self.epoch = 0
            self.G = Generator()
            self.D = Discriminator()
            self.g_optimizer = optim.Adadelta(self.G.parameters())
            self.d_optimizer = optim.Adadelta(self.D.parameters())

        vgg19_layers = list(models.vgg19(pretrained = True).features)[:30]
        self.vgg19 = nn.Sequential(*vgg19_layers)

        self.adversarial_loss = torch.nn.BCELoss()

        self.network.to(device)

    def load_state(self, fname):
        with open(fname, 'rb') as f:
            state = pickle.load(f)
        
        self.epoch      = state["epoch"]
        self.D          = state["D"]
        self.G          = state["G"]
        self.g_optimizer  = state["g_optimizer"]
        self.d_optimizer  = state["d_optimizer"]

    def save_state(self):
        fname = "%s/save" % self.args.save_dir
        state = {
            "D"         : self.D,
            "G"         : self.G,
            "epoch"     : self.epoch,
            "optimizer" : self.optimizer,
        }
        with open("%s_%d.pkl" % (fname, self.epoch), 'wb') as f:
            pickle.dump(state, f)

    def train(self, dataloader):
        while self.epoch < self.args.epochs:
            print("=== Epoch: %d ===" % self.epoch)
            if self.epoch % self.args.save_epochs == 0:
                self.save_state()
            if self.epoch % self.args.eval_epochs == 0:
                self.evaluate()

            self.D.train()
            self.G.train()
            g_losses = []
            d_losses = []

            for low_res, high_res in dataloader:
                batch_size = high_res.size(0)
                real = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

                """ Generator training """
                self.g_optimizer.zero_grad()

                generated = self.G(low_res)

                content_loss = torch.mean(self.vgg19(high_res) - self.vgg19(generated))
                adversarial_loss = torch.sum(-self.D(generated))

                g_loss = content_loss + 1E-3 * adversarial_loss
                g_losses.append(g_loss.item())
                g_loss.backwards()
                self.g_optimizer.step()

                """ Discriminator training """
                self.d_optimizer.zero_grad()

                real_loss = self.adversarial_loss(self.D(high_res), real)
                fake_loss = self.adversarial_loss(self.D(generated.detach()), fake)

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
        self.evaluate()
        sys.stdout.flush()

    def evaluate(self):
        self.network.eval()

        # TODO:

class ResidualBlock(nn.Module):
    def __init__(self):
        res_block = nn.Sequential(
            nn.Conv2D(64, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.PReLU(),
            nn.Conv2D(64, 64, 3, padding=1),
            nn.BatchNorm2D(64),
        )

    def forward(self, x):
        return x + res_block(x)

def shuffle_block():
    return nn.Sequential(
        nn.Conv2D(64, 256, 3, padding=1),
        nn.PixelShuffle(2),
        nn.PReLU()
    )

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.B = 16
        self.S = 2

        self.conv1      = nn.Sequential(nn.Conv2D(3, 64, 9, padding=4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock() for _ in range(self.B)])
        self.conv2      = nn.Sequential(nn.Conv2D(64, 64, 3, padding=1), nn.BatchNorm2D(64))
        self.shuffle    = nn.Sequential(*[shuffle_block() for _ in range(self.S)])
        self.conv3      = nn.Conv2D(64, 3, 9, padding=4)

    def forward(self, x):
        """
        Params:
        - x ([N, 3, H, W] Tensor): batch of images to super-resolve

        Returns:
        - [N, 3, 4H, 4W] Tensor
        """
        x = image.float().to(device)

        x = self.conv1(x)
        saved_x = x
        x = self.res_blocks(x)
        x = self.conv2(x) + saved_x
        x = self.shuffle(x)
        x = self.conv3(x)
        return x

def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2D(out_channels),
        nn.LeakyReLU(0.2)
    )

def conv_size(x_in, kernel_size, stride, padding):
    return floor((x_in + 2 * padding - kernel_size) / stride + 1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        in_channels = 3
        im_width = 96

        layers = []
        for idx, out_channels in enumerate([64, 128, 256, 512]):
            if idx == 0:
                layers.extend([nn.Conv2D(in_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2)])
            else:
                layers.append(conv_block(in_channels, out_channels, 3, 1, 1))

            layers.append(conv_block(out_channels, out_channels, 3, 2, 1))

            im_width = conv_size(im_width, 3, 1, 1)
            im_width = conv_size(im_width, 3, 2, 1)
            in_channels = out_channels

        self.fc = nn.Sequential(
            nn.Linear(im_width * im_width * 512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.model = nn.Sequential(*layers)

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
