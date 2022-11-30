from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import torch
import torch.utils.data as data_utils


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, img_size=64, n_exs=-1):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize(img_size)
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)

        if n_exs != -1:
            self.dataset = data_utils.Subset(self.dataset, torch.arange(n_exs))
            
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CelebDataLoader(BaseDataLoader):
    """
    CelebA data loading
    Download and extract:
    https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip 
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, img_size=64, n_exs=-1):
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])

        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(
            self.data_dir, transform=transform)

        if n_exs != -1:
            self.dataset = data_utils.Subset(self.dataset, torch.arange(n_exs))

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class WIKIDataLoader(BaseDataLoader):
    """
    WIKI data loading
    Download and extract:
    https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, img_size=64, n_exs=-1):
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(
            self.data_dir, transform=transform)

        if n_exs != -1:
            self.dataset = data_utils.Subset(self.dataset, torch.arange(n_exs))

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
