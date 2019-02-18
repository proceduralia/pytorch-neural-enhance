import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
import argparse
import datetime
import os
import random
from datasets import FivekDataset
from models import CAN, SandOCAN, UNet
from torch_utils import JoinedDataLoader, load_model
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda_idx', type=int, default=1, help='cuda device id')
parser.add_argument('--logdir', default='log', help='logdir for tensorboard')
parser.add_argument('--image_path', default=None, help='path to the image to enhance')
parser.add_argument('--run_tag', default='evaluation', help='tags for the current run')
parser.add_argument('--final_dir', default="final_models", help='directory for the final_models')
parser.add_argument('--model_type', default='can32', choices=['can32', 'sandocan32','unet'], help='type of model to use')
parser.add_argument('--data_path', default='/home/iacv3_1/fivek', help='path of the base directory of the dataset')
opt = parser.parse_args()

#Create writer for tensorboard
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
run_name = "{}_{}".format(opt.run_tag,date) if opt.run_tag != '' else date
log_dir_name = os.path.join(opt.logdir, run_name)
writer = SummaryWriter(log_dir_name)
writer.add_text('Options', str(opt), 0)
print(opt)

if torch.cuda.is_available() and not opt.cuda:
	print("You should run with CUDA.")
device = torch.device("cuda:"+str(opt.cuda_idx) if opt.cuda else "cpu")

landscape_transform = transforms.Compose([
                              transforms.Resize((332, 500)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #normalize in [-1,1]
                             ])
portrait_transform = transforms.Compose([
                              transforms.Resize((500, 332)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #normalize in [-1,1]
                             ])
"""
landscape_dataset = FivekDataset(opt.data_path, expert_idx=2, transform=landscape_transform, filter_ratio="landscape")
portrait_dataset = FivekDataset(opt.data_path, expert_idx=2, transform=portrait_transform, filter_ratio="portrait")


landscape_loader = DataLoader(train_landscape_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2)
portrait_loader = DataLoader(train_portrait_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2)
loader = JoinedDataLoader(train_landscape_loader, train_portrait_loader)
"""

im = Image.open(opt.image_path)
if im.size()[2] > im.size()[1]:
	im = landscape_transform(im)
else:
	im = portrait_transform(im)

im = im.to(device)
im = im.unsqueeze(0)

if opt.model_type == 'can32':
  model = CAN(n_channels=32)
if opt.model_type == 'sandocan32':
  model = SandOCAN()
if opt.model_type == 'unet':
  model = UNet()
assert model

models_path = [f for f in os.listdir(opt.final_dir) if f.startswith(opt.model_type)]
#print(models_path)
images = im
for model_name in models_path:
	print('Loading model' + model_name)
	model.load_state_dict(torch.load(os.path.join(opt.final_dir,model_name), map_location=lambda storage, loc: storage))
	model.to(device)
	images = torch.cat((images,model(im)))
names = [' '.join(m.split('_')[1:-1]) for m in models_path]
filename = opt.model_type+'actual_'+'_'.join(names)+'.png'
save_image(images,filename,normalize=True,range=(-1,1))
grid = make_grid(images, nrow=1, normalize=True, range=(-1,1))
writer.add_image(filename, grid)
