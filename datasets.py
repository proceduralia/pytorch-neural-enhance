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
    def __init__(self, transformation, root='./data', train=True, normalize=True):
        """Args:
            transformation (callable): the skimage-style transformation to be applied
            root (str): the root of the original cifar data
            train (bool): true for training set, false for test set
            normalize (bool): if true, normalize the data in [-1,1]
        """
        if train:
            data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True).train_data
            gray_data = np.array([rgb2gray(im) for im in data])
            self.original_data = torch.FloatTensor(gray_data)
            self.transformed_data = torch.FloatTensor(np.array([transformation(im) for im in gray_data]))
        else:
            data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True).test_data
            gray_data = np.array([rgb2gray(im) for im in data])
            self.original_data = torch.FloatTensor(gray_data)
            self.transformed_data = torch.FloatTensor(np.array([transformation(im) for im in gray_data]))
        
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
