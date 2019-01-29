from torch.utils.data import Dataset
import torch
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
from skimage.color import rgb2gray
from skimage.transform import resize  
from skimage.exposure import equalize_hist, equalize_adapthist

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

    def __getitem__(self, i):
        return self.original_data[i], self.transformed_data[i]

    def __len__(self):
        return len(self.original_data)

if __name__ == "__main__":
    dataset = TransformedCifarDataset(equalize_hist)
    print(len(dataset))
    print(dataset[0])
