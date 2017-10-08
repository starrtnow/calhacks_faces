import torchvision
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torch.autograd import Variable
from networks import VAENetwork
import time
import argparse

if __name__ == "__main__":
    params = {
        'learning rate': 0.005,
        'encoder dropout': 0.65,
        'decoder dropout': 0.45,
        'z_channel': 32,
        'z_dim': 128
    }
    parser = argparse.ArgumentParser("Trains")
    parser.add_argument("-e", "--epoch", action='store', type=int, default = "200")
    parser.add_argument("-r", "--root", action='store', default="data/faces")
    parser.add_argument("-s", "--save", action='store', type=int, default="50")
    parser.add_argument("-n", "--network", action='store', default="networks")
    parser.add_argument("-l", "--load", action='store', default="none")
    args = parser.parse_args()

    n_epochs = args.epoch
    dataset = torchvision.datasets.ImageFolder(root=args.root, transform=tf.ToTensor())
    net = VAENetwork(dataset, params, False)
    if args.load != "none":
        net.load(args.load)
        print("Loaded {}".format(args.load))

    for epoch in range(1, n_epochs + 1):
        result = net.train_epoch(epoch)
        current_time = time.strftime("[%d-%m-%Y %H-%M-%S]")
        print(result)
        if epoch % args.save == 0:
            sample = net.sample()
            torchvision.utils.save_image(sample, "sample/{}.png".format(current_time))

    net.save(args.network)
    print("Saved {}".format(args.network))



