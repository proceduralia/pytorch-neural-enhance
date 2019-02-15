import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import numpy as np
import argparse
from functools import partial
import datetime
import os
import random
from datasets import TransformedCifarDataset
from skimage.exposure import equalize_hist, equalize_adapthist
from transforms import unsharp_mask
from scipy.stats import wasserstein_distance
from models import MLP, NaiveCNN, LittleUnet

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda_idx', type=int, default=1, help='cuda device id')
parser.add_argument('--outf', default='.', help='folder for model checkpoints')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--logdir', default='log_histeq', help='logdir for tensorboard')
parser.add_argument('--run_tag', default='', help='tags for the current run')
parser.add_argument('--checkpoint_every', default=10, help='number of epochs after which saving checkpoints')
parser.add_argument('--model_type', default='unet', choices=['unet', 'cnn', 'mlp'], help='type of model to use')
parser.add_argument('--loss', default='mse', choices=['mse','mae'], help='type of loss to use')
parser.add_argument('--initial_1by1', action="store_true", help='whether to use the initial 1 by 1 convs in unet')
parser.add_argument('--transform', default='ad_hist_eq', choices=['hist_eq','ad_hist_eq','unsharp'], help='transformation to be learned')
opt = parser.parse_args()

#Create writer for tensorboard
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
run_name = "{}_{}".format(opt.run_tag,date) if opt.run_tag != '' else date
log_dir_name = os.path.join(opt.logdir, run_name)
writer = SummaryWriter(log_dir_name)
writer.add_text('Options', str(opt), 0)
print(opt)

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
    
if torch.cuda.is_available() and not opt.cuda:
	print("You should run with CUDA.")
device = torch.device("cuda:"+str(opt.cuda_idx) if opt.cuda else "cpu")

transforms = {
    'hist_eq': equalize_hist,
    'ad_hist_eq': partial(equalize_adapthist, kernel_size=32//4),
    'unsharp': partial(unsharp_mask, amount=1.0)
}

dataset = TransformedCifarDataset(transforms[opt.transform])
loader = DataLoader(dataset, batch_size=opt.batch_size,
            shuffle=True, num_workers=2)
test_dataset = TransformedCifarDataset(transforms[opt.transform], train=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batch_size,
            shuffle=False, num_workers=2)

if opt.model_type == 'unet':
  model = LittleUnet(initial_1by1=opt.initial_1by1)
if opt.model_type == 'cnn':
  model = NaiveCNN()
if opt.model_type == 'mlp':
  model = MLP()
assert model

model = model.to(device)
if opt.loss == "mse":
  criterion = nn.MSELoss()
if opt.loss == "mae":
  criterion = nn.L1Loss()

criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

for epoch in range(opt.epochs):
    model.train()
    cumulative_loss = 0.0
    for i, (im_o, im_t) in enumerate(loader):
        im_o, im_t = im_o.to(device), im_t.to(device)
        optimizer.zero_grad()
        
        output = model(im_o)
        loss = criterion(output, im_t)
        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item()
        print('[Epoch %d, Batch %2d] loss: %.3f' %
         (epoch + 1, i + 1, cumulative_loss / (i+1)), end="\r")
    #Evaluate 
    writer.add_scalar('MSE Train', cumulative_loss / len(loader), epoch)
    model.eval()
    
    test_loss = []
    wass_dist = []
    for i, (im_o, im_t) in enumerate(test_loader): 
      im_o, im_t = im_o.to(device), im_t.to(device)
      with torch.no_grad():
        output = model(im_o)
        test_loss.append(criterion(output, im_t).item())
        actual_hists = np.array([np.histogram(im, bins=255, density=True)[0] for im in im_t.cpu().numpy()]) 
        pred_hists = np.array([np.histogram(pred, bins=255, density=True)[0] for pred in output.cpu().numpy()])
        wass_dist.append(np.mean([wasserstein_distance(i, j) for i,j in zip(actual_hists, pred_hists)]))
    writer.add_scalar('MSE Test', sum(test_loss)/len(test_loss), epoch)
    writer.add_scalar('Avg Wasserstein distance', sum(wass_dist)/len(wass_dist), epoch)
    
    #Make list of type [original1,estimated1,actual1,original2,estimated2,actual2]
    original, actual = test_dataset[:5]
    original, actual = original.to(device), actual.to(device)
    estimated = model(original)
    #Original, tran and estimated are (5, 1, 32, 32)
    images = [[o,e,a] for o,e,a in zip(original,estimated,actual)]
    images = torch.cat([i for k in images for i in k]).unsqueeze(1)
    #Make a grid, in each row, original|estimated|actual
    grid = make_grid(images, nrow=len(images)//5, normalize=True)
    writer.add_image('Original|Estimated|Actual', grid, epoch)
