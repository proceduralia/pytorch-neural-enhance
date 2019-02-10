from tensorboardX import SummaryWriter 
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import torchvision
import torchvision.transforms as transforms
from skimage.color import rgb2gray
from skimage.transform import resize  
from skimage.exposure import equalize_hist, equalize_adapthist
from PIL import Image
import pandas as pd

class TransformedCifarDataset(Dataset):
    """
     Dataset with (x, transformed_x) couples, given CIFAR10 and a skimage-style transformation
    """
    def __init__(self, transformation, root='./data', tag='adhist', train=True, normalize=True):
      """Args:
          transformation (callable): the skimage-style transformation to be applied
          root (str): the root of the original cifar data
          train (bool): true for training set, false for test set
          normalize (bool): if true, normalize the data in [-1,1]
      """
      original_train_path = os.path.join(root, "original_train"+tag+'.pt')
      transformed_train_path = os.path.join(root, "transformed_train"+tag+'.pt')
      original_test_path = os.path.join(root, "original_test"+tag+'.pt')
      transformed_test_path = os.path.join(root, "transformed_test"+tag+'.pt')
      
      if train:
        if os.path.exists(original_train_path):
          self.original_data = torch.load(original_train_path)
          self.transformed_data = torch.load(transformed_train_path)
        else:
          data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True).train_data
          gray_data = np.array([rgb2gray(im) for im in data])
          self.original_data = torch.FloatTensor(gray_data)
          self.transformed_data = torch.FloatTensor(np.array([transformation(im) for im in gray_data]))
          torch.save(self.original_data, original_train_path)
          torch.save(self.transformed_data, transformed_train_path)

      if not train:
        if os.path.exists(original_test_path):
          self.original_data = torch.load(original_test_path)
          self.transformed_data = torch.load(transformed_test_path)
        else:
          data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True).test_data
          gray_data = np.array([rgb2gray(im) for im in data])
          self.original_data = torch.FloatTensor(gray_data)
          self.transformed_data = torch.FloatTensor(np.array([transformation(im) for im in gray_data]))
          torch.save(self.original_data, original_test_path)
          torch.save(self.transformed_data, transformed_test_path)
   
        if normalize:
            normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            self.original_data = normalization(self.original_data)
            self.transformed_data = normalization(self.transformed_data)
      self.original_data = self.original_data.unsqueeze(1)
      self.transformed_data = self.transformed_data.unsqueeze(1)

    def __getitem__(self, idx):
        return self.original_data[idx], self.transformed_data[idx]

    def __len__(self):
        return len(self.original_data)
        
class FivekDataset(Dataset):
    def __init__(self, base_path, expert_idx=2, transform=None, filter_ratio=None, use_features=False):
      """Fivek dataset class.
      Args:
        - base_path (str): base path with the directories
        - expert_idx (int): index of the ground truth expert
        - transform (torchvision transform): to be applied to both original and improved images
        - filter_ratio (str): "landscape" or "portrait" filter
        - use_features (bool): whether to use the (subject, light, location, time) features or not
      """
      self.base_path = base_path
      self.expert_idx = expert_idx
      self.use_features = use_features
      if use_features:
        self.info_df = pd.read_csv(os.path.join(base_path, 'mitdatainfo.csv'))
        self.features = ["subject", "light", "location", "time"]
        self.encoders = {}
        for feature_name in self.features:
          self.encoders[feature_name] = LabelEncoder().fit(self.info_df[feature_name])
        self.encoded_features = torch.LongTensor(np.vstack([self.encoders[feat].transform(self.info_df[feat]) for feat in self.features]).T)
  
      self.transform = transform
      if filter_ratio:
        assert filter_ratio in ["landscape", "portrait"]
      self.filter_ratio = filter_ratio
      self.original_path = os.path.join(base_path, 'original')
      self.expert_path = os.path.join(base_path, 'expert'+str(expert_idx))
      
      self.len = len(os.listdir(self.original_path))
      #TODO inefficient... Just save this data in the csv
      original_shapes = []
      for i in range(self.len):
        original_shapes.append(Image.open(os.path.join(self.original_path, "{}.png".format(i))).size)
      self.landscape_idxs = [i for i in range(len(original_shapes)) if original_shapes[i][0] > original_shapes[i][1]]
      self.portrait_idxs = [i for i in range(len(original_shapes)) if original_shapes[i][0] < original_shapes[i][1]]
      
    def __getitem__(self, idx):
      #Alter index if poltrait or landscape filter is selected
      idx = int(idx)
      if self.filter_ratio == "landscape":
        idx = self.landscape_idxs[idx]
      if self.filter_ratio == "portrait":
        idx = self.portrait_idxs[idx]
      original_im = Image.open(os.path.join(self.original_path, str(idx)+'.png'))
      expert_im = Image.open(os.path.join(self.expert_path, str(idx)+'.png'))
      if self.transform:
        original_im = self.transform(original_im)
        expert_im = self.transform(expert_im)
      if self.use_features:
        #Retrieve features from dataframe and transform them
        feats = self.encoded_features[idx]
        #Create tuple of tensors
        return original_im, expert_im, tuple([tens for tens in feats])
      else:  
        return original_im, expert_im

    def __len__(self):
      if self.filter_ratio == "landscape":
        return len(self.landscape_idxs)
      if self.filter_ratio == "portrait":
        return len(self.portrait_idxs)
      else:
        return self.len

if __name__ == "__main__":
    dataset = FivekDataset(base_path="/home/iacv3_1/fivek", use_features=True)
    original_im, expert_im, feats = dataset[0]
    print(original_im.size, expert_im.size, feats)
