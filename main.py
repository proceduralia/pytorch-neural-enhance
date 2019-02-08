import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import argparse
import datetime
import os
import random
from datasets import FivekDataset
from models import CAN
from torch_utils import JoinedDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda_idx', type=int, default=1, help='cuda device id')
parser.add_argument('--outf', default='.', help='folder for model checkpoints')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--logdir', default='log', help='logdir for tensorboard')
parser.add_argument('--run_tag', default='', help='tags for the current run')
parser.add_argument('--checkpoint_every', default=10, help='number of epochs after which saving checkpoints')
parser.add_argument('--model_type', default='can32', choices=['can32'], help='type of model to use')
parser.add_argument('--data_path', default='/home/iacv3_1/fivek', help='path of the base directory of the dataset')
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

landscape_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                              transforms.Resize((332, 500)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #normalize in [-1,1]
                             ])
portrait_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                              transforms.Resize((500, 332)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #normalize in [-1,1]
                             ])
landscape_dataset = FivekDataset(opt.data_path, expert_idx=2, transform=landscape_transform, filter_ratio="landscape")
portrait_dataset = FivekDataset(opt.data_path, expert_idx=2, transform=portrait_transform, filter_ratio="portrait")


train_size = int(0.8 * len(landscape_dataset))
test_size = len(landscape_dataset) - train_size
train_landscape_dataset, test_landscape_dataset = random_split(landscape_dataset, [train_size, test_size])

train_size = int(0.8 * len(portrait_dataset))
test_size = len(portrait_dataset) - train_size
train_portrait_dataset, test_portrait_dataset = random_split(portrait_dataset, [train_size, test_size])

train_loader = JoinedDataLoader(
               DataLoader(train_landscape_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2),
               DataLoader(train_portrait_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
               )
test_loader = JoinedDataLoader(
               DataLoader(test_landscape_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2),
               DataLoader(test_portrait_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
               )

if opt.model_type == 'can32':
  model = CAN(n_channels=32)
assert model

model = model.to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

for epoch in range(opt.epochs):
    model.train()
    cumulative_loss = 0.0
    for i, (im_o, im_t) in enumerate(train_loader):
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
    #Checkpointing
    if epoch % opt.checkpoint_every == 0:
        torch.save(model.state_dict(), "{}_epoch{}.pt".format(opt.run_tag, epoch) 
    model.eval()
    
    test_loss = []
    wass_dist = []
    for i, (im_o, im_t) in enumerate(test_loader): 
      im_o, im_t = im_o.to(device), im_t.to(device)
      with torch.no_grad():
        output = model(im_o)
        test_loss.append(criterion(output, im_t).item())
    writer.add_scalar('MSE Test', sum(test_loss)/len(test_loss), epoch)
    
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
