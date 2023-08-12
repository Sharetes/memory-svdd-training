from __future__ import print_function
import os
import sys
import torch
from torch.utils.data import Subset, random_split
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import lightning as L
from random import sample
import random

from torchvision.datasets import CIFAR10
# from .preprocessing import global_contrast_normalization,get_target_label_idx
import numpy as np

# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.getcwd())
from utils.plot_images_grid import plot_images_grid
from preprocessing import global_contrast_normalization, get_target_label_idx


class CIFAR10DataModel(L.LightningDataModule):
    def __init__(self,
                 batch_size,
                 normal_class,
                 radio=0,
                 num_workers=8,
                 root="./data/",
                 dataset_name="cifar10"):
        super().__init__()
        # normal class only one class per training set
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        # self.prepare_data_per_node = Falseself.center = self.center.to(self.device)
        self.root = root
        # 污染数据比例
        self.radio = radio
        self.normal_class = normal_class
        self.num_workers = num_workers
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

    # def prepare_data(self):
    #     # download
    #     CIFAR10(self.root, train=True, download=True)
    #     CIFAR10(self.root, train=False, download=True)

    def setup(self, stage: str) -> None:

        # Pre-computed min and max values (after applying GCN) from train data per class
        # global_contrast_normalization
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_max[self.normal_class][0]] * 3, [
                min_max[self.normal_class][1] - min_max[self.normal_class][0]
            ] * 3)
        ])
        target_transform = transforms.Lambda(
            lambda x: int(x in self.outlier_classes))

        if stage == "fit":
            train_cifar10 = CIFAR10(root=self.root,
                                    train=True,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=False)
            train_indices = [
                idx for idx, target in enumerate(train_cifar10.targets)
                if target in self.normal_classes
            ]
            dirty_indices = [
                idx for idx, target in enumerate(train_cifar10.targets)
                if target not in self.normal_classes
            ]
            train_indices += sample(
                dirty_indices,
                int(len(train_indices) * self.radio / (1 - self.radio)))
            # 随机打乱正常数据和污染数据索引
            random.shuffle(train_indices)
            # extract the normal class of cifar10 train dataset
            self.train_cifar10 = Subset(train_cifar10, train_indices)

        # if stage == "test":
        self.test_cifar10 = CIFAR10(root=self.root,
                                    train=False,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=False)

    def train_dataloader(self):
        return DataLoader(self.train_cifar10,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_cifar10,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_cifar10,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)


class CIFAR10DataModelpl(pl.LightningDataModule):
    def __init__(self,
                 batch_size,
                 normal_class,
                 radio=0,
                 num_workers=8,
                 root="./data/",
                 dataset_name="cifar10"):
        super().__init__()
        # normal class only one class per training set
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        # self.prepare_data_per_node = Falseself.center = self.center.to(self.device)
        self.root = root
        # 污染数据比例
        self.radio = radio
        self.normal_class = normal_class
        self.num_workers = num_workers
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

    # def prepare_data(self):
    #     # download
    #     CIFAR10(self.root, train=True, download=True)
    #     CIFAR10(self.root, train=False, download=True)

    def setup(self, stage: str) -> None:

        # Pre-computed min and max values (after applying GCN) from train data per class
        # global_contrast_normalization
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_max[self.normal_class][0]] * 3, [
                min_max[self.normal_class][1] - min_max[self.normal_class][0]
            ] * 3)
        ])
        target_transform = transforms.Lambda(
            lambda x: int(x in self.outlier_classes))

        if stage == "fit":
            train_cifar10 = CIFAR10(root=self.root,
                                    train=True,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=False)
            train_indices = [
                idx for idx, target in enumerate(train_cifar10.targets)
                if target in self.normal_classes
            ]
            dirty_indices = [
                idx for idx, target in enumerate(train_cifar10.targets)
                if target not in self.normal_classes
            ]
            train_indices += sample(
                dirty_indices,
                int(len(train_indices) * self.radio / (1 - self.radio)))
            # 随机打乱正常数据和污染数据索引
            random.shuffle(train_indices)
            # extract the normal class of cifar10 train dataset
            self.train_cifar10 = Subset(train_cifar10, train_indices)

        # if stage == "test":
        self.test_cifar10 = CIFAR10(root=self.root,
                                    train=False,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=False)

    def train_dataloader(self):
        return DataLoader(self.train_cifar10,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_cifar10,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_cifar10,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)


def get_gcn():
    os.makedirs("log/cifar10", exist_ok=True)

    print(os.path.dirname(__file__))
    print(os.path.dirname(os.path.dirname(__file__)))
    # i = 0
    # for inputs, labels in data_loader_train:
    #     print(inputs.shape)
    #     # plot_images_grid(inputs, export_img="log/cifar10/train_%d" % i)
    #     break
    #     i += 1
    train_set_full = CIFAR10(
        root="./data/",
        train=True,
        #  download=True,
        transform=None,
        target_transform=None)

    MIN = []
    MAX = []
    for normal_classes in range(10):
        train_idx_normal = get_target_label_idx(train_set_full.targets,
                                                normal_classes)
        train_set = Subset(train_set_full, train_idx_normal)

        _min_ = []
        _max_ = []
        for idx in train_set.indices:
            # print(train_set.dataset.data[idx])
            gcm = global_contrast_normalization(
                torch.from_numpy(train_set.dataset.data[idx]).double(), 'l1')
            _min_.append(gcm.min())
            _max_.append(gcm.max())
        MIN.append(np.min(_min_))
        MAX.append(np.max(_max_))
    print(list(zip(MIN, MAX)))


if __name__ == '__main__':
    cifar10 = CIFAR10DataModel(batch_size=100, normal_class=1, radio=0.1)
    cifar10.setup("fit")
