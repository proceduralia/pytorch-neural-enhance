import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import argparse
from functools import partial
from datasets import TransformedCifarDataset
from skimage.exposure import equalize_hist, equalize_adapthist
from transforms import unsharp_mask
from scipy.stats import wasserstein_distance
from models import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='.', help='folder for model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--logdir', default='log', help='logdir for tensorboard')
parser.add_argument('--run_tag', default='', help='tags for the current run')
parser.add_argument('--checkpoint_every', default=10, help='number of epochs after which saving checkpoints') 
parser.add_argument('--model_type', default='mlp', help='type of model to use')
parser.add_argument('--transform', default='hist_eq', choices=['hist_eq','ad_hist_eq','unsharp'], help='transformation to be learned')
opt = parser.parse_args()

transforms = {
    'hist_eq': equalize_hist,
    'ad_hist_eq': partial(equalize_adapthist, kernel_size=32//4),
    'unsharp': partial(unsharp_mask, amount=1.0)
}

dataset = TransformedCifarDataset(transforms[opt.transform])
loader = DataLoader(dataset, batch_size=opt.batch_size,
            shuffle=True, num_workers=2)

model = MLP()
model.train()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

for epoch in range(opt.epochs):
    running_loss = 0.0
    for i, (im_o, im_t) in enumerate(loader):
        optimizer.zero_grad()
	
        output = model(im_o)
        loss = criterion(output, im_t)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

#TODO add evaluation, model checkpointing, tensorboard
