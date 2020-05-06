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

def log_message(message):
    print("{} │ {}".format(time.strftime('%l:%M%p'), message))

class Model():
    def __init__(self, args):
        self.args = args

        self.pretrained = False
        self.epoch = 0
        self.G = Generator()
        self.D = Discriminator()
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=1E-4)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=1E-4)
        self.train_losses = []
        self.val_losses = []

        if args.load_model:
            self._load_state(args.load_model)

        # extract all layers prior to the last softmax of VGG-19
        vgg19_layers = list(models.vgg19(pretrained = True).features)[:36]
        self.vgg19 = nn.Sequential(*vgg19_layers).eval()
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
            log_message("Starting pretraining")
            self._pretrain(train_dataloader)
            self._save_state()

            if val_dataloader:
                val_g_loss, _ = self.evaluate(val_dataloader)
                log_message("Pretrain G loss: {:.4f}".format(val_g_loss))

        """ Real Training """
        log_message("Starting training")
        while self.epoch < self.args.epochs:
            # Train one epoch
            self.D.train()
            self.G.train()
            g_loss, d_loss = self._run_epoch(train_dataloader, train=True)
            self.train_losses.append([g_loss, d_loss])
            self.epoch += 1
            log_message("Epoch: {}/{}".format(self.epoch, self.args.epochs))

            # Print evaluation
            train_string = "Train G loss: {:.4f} | Train D loss: {:.4f}".format(g_loss, d_loss)
            if self.epoch % self.args.eval_epochs == 0:
                if val_dataloader:
                    val_g_loss, val_d_loss = self.evaluate(val_dataloader)
                    self.val_losses.append([val_g_loss, val_d_loss])
                    train_string += " | Val G loss: {:.4f} | Val D loss: {:.4f}".format(val_g_loss, val_d_loss)
            log_message(train_string)

            # Save the model
            if self.epoch % self.args.save_epochs == 0:
                self._save_state()

        log_message("Finished training")
        self._save_state()

    def evaluate(self, dataloader):
        self.D.eval()
        self.G.eval()

        with torch.no_grad():
            return self._run_epoch(dataloader, train=False)

    def generate(self, dataloader):
        def to_image(tensor):
            array = tensor.data.cpu().numpy()
            array = array.transpose((1, 2, 0))
            array = np.clip(255.0 * array, 0, 255)
            array = np.uint8(array)
            return Image.fromarray(array)

        self.D.eval()
        self.G.eval()

        if not os.path.exists(self.args.generate_dir):
            os.mkdir(self.args.generate_dir)

        with torch.no_grad():
            idx = 0
            for batch in dataloader:
                low_res  = batch['low_res'].to(device)
                hi_res  = batch['high_res']
                generated = self.G(low_res)

                for i in range(len(generated)):
                    fake_im = to_image(generated[i])
                    real_im = to_image(hi_res[i])

                    fake_im.save(os.path.join(self.args.generate_dir, "{}_fake.png".format(i)))
                    real_im.save(os.path.join(self.args.generate_dir, "{}_real.png".format(i)))

    def _load_state(self, fname):
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'
        state = torch.load(fname, map_location=map_location)

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
            log_message("Pretrain Epoch: {}/{}".format(i, self.args.pretrain_epochs))
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


            """ Discriminator """
            generated = self.G(low_res)
            self.d_optimizer.zero_grad()

            real_loss = self.bce_loss(self.D(high_res), real)
            fake_loss = self.bce_loss(self.D(generated), fake)
            d_loss = real_loss + fake_loss
            d_losses.append(d_loss.item())

            if train:
                d_loss.backward()
                self.d_optimizer.step()

            """ Generator """
            generated = self.G(low_res)
            self.g_optimizer.zero_grad()

            pixel_loss = self.mse_loss(high_res, generated)
            content_loss = self.mse_loss(self.vgg19((high_res + 1) / 2), self.vgg19((generated + 1) / 2))
            adversarial_loss = self.bce_loss(self.D(generated), real)
            g_loss = pixel_loss + 0.006 * content_loss + 1E-3 * adversarial_loss
            g_losses.append(g_loss.item())

            if train:
                g_loss.backward()
                self.g_optimizer.step()

        return np.mean(g_losses), np.mean(d_losses)
