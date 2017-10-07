import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from networks import Network, Result
from torch.utils.data import DataLoader

from utils import conv2dsize


class Encoder(nn.Module):
    def __init__(self, side_length, n_channel, z_channel, z_dim, dropout = 0.5, use_cuda = False):
        super(Encoder, self).__init__()

        '''
        side_length is the length in pixels of the square input image
        n_channel is the number of channels (3 for RGB, 1 for grayscale)
        z_channel is the quantized number of output channels that the convolution use
        z_dim is the size of the latent vector
        '''

        self.n_channel = n_channel
        self.z_channel = z_channel
        self.z_dim = z_dim
        self.use_cuda = use_cuda

        kernel_size = 4


        self.out_dimension = conv2dsize(32, kernel_size, 2, 0)
        for i in range(3):
            self.out_dimension = conv2dsize(self.out_dimension, kernel_size, 2, 1)

        '''
        Networks calculate mu and the log var in order to use the reparameterization trick
        '''
        self.mu = nn.Sequential(
            nn.Conv2d(n_channel, z_channel, kernel_size, 2, 0, bias=False),
            nn.BatchNorm2d(z_channel),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel, z_channel * 2, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(z_channel * 2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel * 2, z_channel * 4, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(z_channel * 4),
            nn.LeakyReLU(),
            nn.Conv2d(z_channel * 4, z_channel * 4, kernel_size, 2, 1, bias=False),
            nn.LeakyReLU()
        )

        self.var = nn.Sequential(
            nn.Conv2d(n_channel, z_channel, kernel_size, 2, 0, bias=False),
            nn.BatchNorm2d(z_channel),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel, z_channel * 2, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(z_channel * 2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel * 2, z_channel * 4, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(z_channel * 4),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel * 4, z_channel * 4, kernel_size, 2, 1, bias=False),
            nn.LeakyReLU()
        )

        self.mu_out = nn.Linear(z_channel * 4 * self.out_dimension, z_dim)
        self.var_out = nn.Linear(z_channel * 4 * self.out_dimension, z_dim)


    def forward(self, X):

        mu_out = self.mu(X)
        mu_out = mu_out.view(mu_out.size(0), -1)
        mu_out = self.mu_out(mu_out)

        var_out = self.var(X)
        var_out = var_out.view(var_out.size(0), -1)
        var_out = self.var_out(var_out)

        return mu_out, var_out

    def sample(self, batch_size, mu, var):
        eps = Variable(torch.randn(batch_size, self.z_dim))
        if self.use_cuda:
            eps = eps.cuda()
        return mu + torch.exp(var/2) * eps

class Decoder(nn.Module):

    def __init__(self, side_length, n_channel, z_channel, z_dim, droppout = 0.5):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.n_channel = n_channel
        self.side_length = side_length

        kernel_size = 4

        self.convnet = nn.Sequential(
            nn.ConvTranspose2d(z_dim, z_channel * 4, kernel_size, 1, 0, bias=False),
            nn.BatchNorm2d(z_channel * 4),
            nn.ReLU(),
            nn.Dropout2d(p=droppout),
            nn.ConvTranspose2d(z_channel * 4, z_channel * 2, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(z_channel * 2),
            nn.ReLU(),
            nn.Dropout2d(p=droppout),
            nn.ConvTranspose2d(z_channel * 2, z_channel, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(z_channel),
            nn.ReLU(),
            nn.Dropout2d(p=droppout),
            nn.ConvTranspose2d(z_channel, n_channel, kernel_size, 2, 1, bias=False),
            nn.Sigmoid()
        )



    def forward(self, X):
        out = X.view(X.size(0), self.z_dim, 1, 1)
        out = self.convnet(out)
        return out


default_params = {
    'z_channel': 16,
    'z_dim': 256,
    'encoder dropout': 0.5,
    'decoder dropout': 0.3,
    'batch size': 200,
    'epochs': 20,
    'learning rate': 0.01
}

class VAENetwork(Network):

    def __init__(self, dataset, hyper_params = {}, cuda=False):
        side_length =32
        RGB = 3


        self.use_cuda = cuda
        self.params = default_params
        self.params.update(hyper_params)

        self.save_path = "saved_networks/VAE"

        self.dataloader = DataLoader(dataset=dataset,
                                     shuffle=True,
                                     batch_size=self.params['batch size'],
                                     drop_last=True)

        self.encoder = Encoder(side_length,
                               RGB,
                               self.params['z_channel'],
                               self.params['z_dim'],
                               self.params['encoder dropout'],
                               self.use_cuda)
        self.decoder =Decoder(side_length,
                              RGB,
                              self.params['z_channel'],
                              self.params['z_dim'],
                              self.params['decoder dropout'])

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(parameters, self.params['learning rate'])

        if self.use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

    def train_epoch(self, epoch = 0):

        TINY = 1e-15 #BCE is NaN with an input of 0
        before_time = time.clock()

        self.encoder.train()
        self.decoder.eval()

        loss_sum = 0
        total = 0
        for img, label in self.dataloader:
            self.optimizer.zero_grad()

            X = Variable(img)
            if self.use_cuda:
                X = X.cuda()

            z_mu, z_var = self.encoder(X)
            z = self.encoder.sample(self.params['batch size'], z_mu, z_var)

            X_reconstructed = self.decoder(z)
            reconstruction_loss = F.binary_cross_entropy(X_reconstructed + TINY, X + TINY, size_average=False)
            KL_loss = z_mu.pow(2).add_(z_var.exp()).mul_(-1).add_(1).add_(z_var)
            KL_loss = torch.sum(KL_loss).mul_(-0.5)
            total_loss = reconstruction_loss + KL_loss
            total_loss.backward()
            self.optimizer.step()

        duration = time.clock() - before_time

        def loss_reporting(loss):
            return "Loss {}".format(loss.data[0])

        report = Result(duration, total_loss, epoch, loss_reporting)
        #TODO: Structured way to return training results
        return report

    def sample(self, *img):
        self.encoder.eval()
        self.decoder.eval()

        if len(img) < 1:
            z = Variable(torch.FloatTensor(self.params['batch size'], self.params['z_dim']).normal_())
            if self.use_cuda:
                z = z.cuda()
        else:
            image = img[0]
            if self.use_cuda:
                image = image.cuda()
            image = Variable(image)
            mu, var = self.encoder(image)
            z = self.encoder.sample(image.size(0), mu, var)

        result = self.decoder(z)
        return result.data.cpu()

    def save(self, name):
        path = os.path.join(self.save_path, name)
        if not os.path.exists(path):
            os.makedirs(path)

        encoder_path = os.path.join(path, "encoder.net")
        decoder_path = os.path.join(path, "decoder.net")
        info_path = os.path.join(path, "info.txt")

        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        with open(info_path, 'w') as f:
            f.write("{}".format(self))

    def load(self, name):
        path = os.path.join(self.save_path, name)
        if not os.path.exists(path):
            raise RuntimeError("Saved network does not exist")

        encoder_path = os.path.join(path, "encoder.net")
        decoder_path = os.path.join(path, "decoder.net")

        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))


    def __str__(self):
        return "Params: \n {} \n Networks: \n {} \n {}".format(self.params, self.encoder, self.decoder)





