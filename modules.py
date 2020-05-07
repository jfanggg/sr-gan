import math
import torch
import torch.nn as nn

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

        # map LR ([0, 1]) to HR ([-1, 1])
        return 2 * x - 1

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
