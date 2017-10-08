import torchvision
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

dataset = torchvision.datasets.ImageFolder(root="data/faces", transform=tf.ToTensor())

params = {
    'learning rate': 0.005,
    'encoder dropout': 0.65,
    'decoder dropout': 0.45,
    'z_channel': 32,
    'z_dim': 128
}

from networks import VAENetwork
import time

net = VAENetwork(dataset, hyper_params=params, cuda=False)
print("Bye")
net.load("mc9")
from scipy import misc
from PIL import Image
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
nithin = pil_loader("starr.jpg")
print("Bonjour")
nithin = tf.ToTensor()(nithin)
nithin = nithin.view(1, 3, 128, 128)
loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)
diter = iter(loader)
img, label =  diter.next()
img.size()
i = net.sample(nithin)
import numpy as np
print("HHH")
import matplotlib.pyplot as plt
print("YYY")
#%matplotlib inline
def show(img):
    npimg = img.numpy()
    print("Hi")
    a = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    torchvision.utils.save_image(a,'generated.png')
show(i[0])

