from modules import Generator, Discriminator
import numpy as np
import os
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

class Model():
    def __init__(self, args):
        self.args = args

        self.pretrained = False
        self.epoch = 0
        self.G = Generator()
        self.D = Discriminator()
        self.g_optimizer = optim.Adadelta(self.G.parameters())
        self.d_optimizer = optim.Adadelta(self.D.parameters())
        self.train_losses = []
        self.val_losses = []

        if args.load_model:
            self._load_state(args.load_model)

        # extract all layers prior to the last softmax of VGG-19
        vgg19_layers = list(models.vgg19(pretrained = True).features)[:30]
        self.vgg19 = nn.Sequential(*vgg19_layers)
        for param in self.vgg19.parameters():
            param.requires_grad = False

        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCELoss()

    def train(self, train_dataloader, val_dataloader=None):
        self.D.to(device)
        self.G.to(device)
        self.vgg19.to(device)

        """ Pretrain Generator """
        if not self.pretrained:
            print("Starting pretraining | {}".format(time.strftime('%l:%M%p')))
            self._pretrain(train_dataloader)
            self._save_state()

            if val_dataloader:
                val_g_loss, _ = self.evaluate(val_dataloader)
                print("Pretrain G loss: {:.4f}".format(val_g_loss))

        """ Real Training """
        print("Starting training | {}".format(time.strftime('%l:%M%p')))
        while self.epoch < self.args.epochs:
            # Train one epoch
            self.D.train()
            self.G.train()
            g_loss, d_loss = self._run_epoch(train_dataloader, train=True)
            self.train_losses.append([g_loss, d_loss])
            self.epoch += 1
            print("Epoch: {}/{} | {}".format(self.epoch, self.args.epochs, time.strftime('%l:%M%p')))

            # Print evaluation
            train_string = "Train G loss: {:.4f} | Train D loss: {:.4f}".format(g_loss, d_loss)
            if self.epoch % self.args.eval_epochs == 0:
                if val_dataloader:
                    val_g_loss, val_d_loss = self.evaluate(val_dataloader)
                    self.val_losses.append([val_g_loss, val_d_loss])
                    train_string += " | Val G loss: {:.4f} | Val D loss: {:.4f}".format(val_g_loss, val_d_loss)
            print(train_string)

            # Save the model
            if self.epoch % self.args.save_epochs == 0:
                self._save_state()

        print("Finished training")
        self._save_state()

    def evaluate(self, dataloader):
        self.D.eval()
        self.G.eval()

        with torch.no_grad():
            return self._run_epoch(dataloader, train=False)

    def generate(self, dataloader):
        self.D.eval()
        self.G.eval()

        if not os.path.exists(self.args.generate_dir):
            os.mkdir(self.args.generate_dir)

        with torch.no_grad():
            idx = 0
            for batch in dataloader:
                low_res  = batch['low_res'].to(device)
                generated = self.G(low_res).data.cpu().numpy()

                for idx, g in enumerate(generated):
                    g = g.transpose((1, 2, 0))
                    g = np.uint8(g)
                    im = Image.fromarray(g)
                    im.save(os.path.join(self.args.generate_dir, "gen_{}.png".format(idx)))

    def _load_state(self, fname):
        state = torch.load(fname)

        self.pretrained = state["pretrained"]
        self.epoch = state["epoch"]
        self.train_losses = state["train_losses"]
        self.val_losses = state["val_losses"]
        self.G.load_state_dict(state["G"])
        self.D.load_state_dict(state["D"])
        self.g_optimizer.load_state_dict(state["g_optimizer"])
        self.d_optimizer.load_state_dict(state["d_optimizer"])

    def _save_state(self):
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        fname = "%s/save_%d.pkl" % (self.args.save_dir, self.epoch)
        state = {
            "pretrained"    : self.pretrained,
            "epoch"         : self.epoch,
            "G"             : self.G.state_dict(),
            "D"             : self.D.state_dict(),
            "g_optimizer"   : self.g_optimizer.state_dict(),
            "d_optimizer"   : self.d_optimizer.state_dict(),
            "train_losses"  : self.train_losses,
            "val_losses"    : self.val_losses
        }
        torch.save(state, fname)

    def _pretrain(self, dataloader):
        self.G.train()
        for i in range(self.args.pretrain_epochs):
            print("Pretrain Epoch: {}/{} | {}".format(i, self.args.pretrain_epochs, time.strftime('%l:%M%p')))
            for batch in dataloader:
                low_res  = batch['low_res'].to(device)
                high_res = batch['high_res'].to(device)

                self.g_optimizer.zero_grad()

                generated = self.G(low_res)

                # Optimize pixel loss
                g_loss = self.mse_loss(generated, high_res)
                g_loss.backward()
                self.g_optimizer.step()

        self.pretrained = True

    def _run_epoch(self, dataloader, train):
        g_losses, d_losses = [], []

        for batch in dataloader:
            low_res  = batch['low_res'].to(device)
            high_res = batch['high_res'].to(device)

            batch_size = high_res.size(0)
            real = torch.ones((batch_size, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)

            """ Generator """
            self.g_optimizer.zero_grad()

            generated = self.G(low_res)

            pixel_loss = self.mse_loss(high_res, generated)
            content_loss = self.mse_loss(self.vgg19(high_res), self.vgg19(generated))
            adversarial_loss = self.bce_loss(self.D(generated), real)
            g_loss = pixel_loss + 0.006 * content_loss + 1E-3 * adversarial_loss
            g_losses.append(g_loss.item())

            if train:
                g_loss.backward()
                self.g_optimizer.step()

            """ Discriminator """
            self.d_optimizer.zero_grad()

            real_loss = self.bce_loss(self.D(high_res), real)
            fake_loss = self.bce_loss(self.D(generated.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_losses.append(d_loss.item())

            if train:
                d_loss.backward()
                self.d_optimizer.step()

        return np.mean(g_losses), np.mean(d_losses)
