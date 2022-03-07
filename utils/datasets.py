import os
import numpy as np
import scipy.io
import h5py
from PIL import Image
from PIL import ImageFile
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

# Adapted DomainNet for reasonable class sizes >= 200, left:
domain_net_targets = ['sea_turtle', 'swan', 'zebra', 'submarine', 'saxophone', 'bird',
                      'squirrel', 'teapot', 'tiger', 'flower', 'streetlight', 'whale', 'feather']

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Digit 5
class MNISTDataset(Dataset):
    def __init__(self, split, transform=None, path='./data/digit-5'):
        # Load Data here
        assert split == 'train' or split == 'test', "Split should be string 'train' or 'test'"
        split = True if split == 'train' else False
        self.dataset = torchvision.datasets.MNIST(root=path,
                                                  train=split,
                                                  transform=transforms.Grayscale(
                                                      num_output_channels=3),
                                                  download=True)
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes
        self.transform = transform
        self.name = 'mnist'

    def show_image(self, idx):
        img, label = self.dataset[idx]
        return img

    def __getitem__(self, idx):
        # dataset[0]
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # len(dataset)
        return len(self.targets)


class SVHNDataset(Dataset):
    def __init__(self, split, transform=None, path='./data/digit-5'):
        # Load Data here
        assert split == 'train' or split == 'test', "Split should be string 'train' or 'test'"
        self.dataset = torchvision.datasets.SVHN(root=path,
                                                 split=split,
                                                 transform=None,
                                                 download=True)
        self.targets = self.dataset.labels
        self.classes = np.unique(self.dataset.labels)
        self.transform = transform
        self.name = 'svhn'

    def show_image(self, idx):
        img, label = self.dataset[idx]
        print(self.classes[int(label)])
        return img

    def __getitem__(self, idx):
        # dataset[0]
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # len(dataset)
        return len(self.targets)


class USPSDataset(Dataset):
    def __init__(self, split, transform=None, path='./data/digit-5/usps.h5'):
        # Load Data
        with h5py.File(path, 'r') as hf:
            data = hf.get(split)
            X = data.get('data')[:] * 255
            y = data.get('target')[:]
            X = np.reshape(X, (len(X), 16, 16))
            X = np.array([np.stack((img.astype(np.uint8),)*3, axis=-1)
                          for img in X])  # Making it 3 channel

        self.X = [Image.fromarray(img, mode="RGB") for img in X]
        self.targets = np.array([int(yi) for yi in y])
        self.classes = np.unique(self.targets)
        self.transform = transform
        self.name = 'usps'

    def show_image(self, idx):
        img, label = self.X[idx], self.targets[idx]
        return img

    def __getitem__(self, idx):
        # dataset[0]
        img, label = self.X[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # len(dataset)
        return len(self.targets)


class SYNDataset(Dataset):
    def __init__(self, split, transform=None, path='./data/digit-5/syn_number.mat'):
        # Load Data
        data = scipy.io.loadmat(path)
        X = data.get(f"{split}_data")
        y = data.get(f"{split}_label")

        self.X = [Image.fromarray(img, mode="RGB") for img in X]
        self.targets = np.array([int(label[0]) for label in y])
        self.classes = np.unique(self.targets)
        self.transform = transform
        self.name = 'syn'

    def show_image(self, idx):
        img, label = self.X[idx], self.targets[idx]
        print(self.classes[int(label)])
        return img

    def __getitem__(self, idx):
        # dataset[0]
        img, label = self.X[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # len(dataset)
        return len(self.targets)


class MNISTMDataset(Dataset):
    def __init__(self, split, transform=None, path='./data/digit-5/mnistm_with_label.mat'):
        # Load Data
        data = scipy.io.loadmat(path)
        X = data.get(f"{split}")
        y = data.get(f"label_{split}")

        self.X = [Image.fromarray(img, mode="RGB") for img in X]
        self.targets = np.array(
            [int(np.where(labelmap == 1)[0][0]) for labelmap in y])
        self.classes = np.unique(self.targets)
        self.transform = transform
        self.name = 'mnistm'

    def show_image(self, idx):
        img, label = self.X[idx], self.targets[idx]
        print(self.classes[int(label)])
        return img

    def __getitem__(self, idx):
        # dataset[0]
        img, label = self.X[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # len(dataset)
        return len(self.targets)


class Digit5Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.name = dataset.name
        self.indices = indices
        self.targets = dataset.targets[indices]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


# DomainNet
class ClipartDataset(Dataset):
    def __init__(self, transform=None, root='data/domainnet/multi', selection_txt_path='data/domainnet/txt/clipart_imgs.txt'):
        # Load Data
        with open(selection_txt_path) as f:
            lines = f.readlines()

        self.img_paths = []
        self.label_list = []
        self.classes = []
        self.label_map = {}
        for x in lines:
            path, label_idx = x.split(' ')
            self.img_paths.append(path)

            label_idx = label_idx.strip()
            self.label_list.append(int(label_idx))

            label_name = path.split('/')[1]
            if label_name not in self.classes:
                self.classes.append(label_name)
            self.label_map[label_idx] = label_name

        self.X = [Image.open(os.path.join(root, img_path))
                  for img_path in self.img_paths]
        self.targets = np.array(self.label_list)
        self.transform = transform
        self.name = 'clipart'

    def show_image(self, idx):
        img, label = self.X[idx], self.targets[idx]
        print(self.label_map.get(str(label)))
        return img

    def __getitem__(self, idx):
        # dataset[0]
        img, label = self.X[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # len(dataset)
        return len(self.targets)


class PaintingDataset(Dataset):
    def __init__(self, transform=None, root='data/domainnet/multi', selection_txt_path='data/domainnet/txt/painting_imgs.txt'):
        # Load Data
        with open(selection_txt_path) as f:
            lines = f.readlines()

        self.img_paths = []
        self.label_list = []
        self.classes = []
        self.label_map = {}
        for x in lines:
            path, label_idx = x.split(' ')
            self.img_paths.append(path)

            label_idx = label_idx.strip()
            self.label_list.append(int(label_idx))

            label_name = path.split('/')[1]
            if label_name not in self.classes:
                self.classes.append(label_name)
            self.label_map[label_idx] = label_name

        self.X = [Image.open(os.path.join(root, img_path))
                  for img_path in self.img_paths]
        self.targets = np.array(self.label_list)
        self.transform = transform
        self.name = 'painting'

    def show_image(self, idx):
        img, label = self.X[idx], self.targets[idx]
        print(self.label_map.get(str(label)))
        return img

    def __getitem__(self, idx):
        # dataset[0]
        img, label = self.X[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # len(dataset)
        return len(self.targets)


class RealDataset(Dataset):
    def __init__(self, transform=None, root='data/domainnet/multi', selection_txt_path='data/domainnet/txt/real_imgs.txt'):
        # Load Data
        with open(selection_txt_path) as f:
            lines = f.readlines()

        self.img_paths = []
        self.label_list = []
        self.classes = []
        self.label_map = {}
        for x in lines:
            path, label_idx = x.split(' ')
            self.img_paths.append(path)

            label_idx = label_idx.strip()
            self.label_list.append(int(label_idx))

            label_name = path.split('/')[1]
            if label_name not in self.classes:
                self.classes.append(label_name)
            self.label_map[label_idx] = label_name

        self.X = [Image.open(os.path.join(root, img_path))
                  for img_path in self.img_paths]
        self.targets = np.array(self.label_list)
        self.transform = transform
        self.name = 'real'

    def show_image(self, idx):
        img, label = self.X[idx], self.targets[idx]
        print(self.label_map.get(str(label)))
        return img

    def __getitem__(self, idx):
        # dataset[0]
        img, label = self.X[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # len(dataset)
        return len(self.targets)


class SketchDataset(Dataset):
    def __init__(self, transform=None, root='data/domainnet/multi', selection_txt_path='data/domainnet/txt/sketch_imgs.txt'):
        # Load Data
        with open(selection_txt_path) as f:
            lines = f.readlines()

        self.img_paths = []
        self.label_list = []
        self.classes = []
        self.label_map = {}
        for x in lines:
            path, label_idx = x.split(' ')
            self.img_paths.append(path)

            label_idx = label_idx.strip()
            self.label_list.append(int(label_idx))

            label_name = path.split('/')[1]
            if label_name not in self.classes:
                self.classes.append(label_name)
            self.label_map[label_idx] = label_name

        self.X = [Image.open(os.path.join(root, img_path))
                  for img_path in self.img_paths]
        self.targets = np.array(self.label_list)
        self.transform = transform
        self.name = 'sketch'

    def show_image(self, idx):
        img, label = self.X[idx], self.targets[idx]
        print(self.label_map.get(str(label)))
        return img

    def __getitem__(self, idx):
        # dataset[0]
        img, label = self.X[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # len(dataset)
        return len(self.targets)
