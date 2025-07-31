# =============================================================================
# NeuSpeech Institute, NeuGaze Project
# Copyright (c) 2024 Yiqian Yang
#
# This code is part of the NeuGaze project developed at NeuSpeech Institute.
# Author: Yiqian Yang
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
# International License. To view a copy of this license, visit:
# http://creativecommons.org/licenses/by-nc/4.0/
# =============================================================================

import os
import numpy as np
import cv2
import json

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


class GazeDataset(Dataset):
    def __init__(self, images, labels,screen_size, transform=None):
        self.images = images
        self.labels = labels
        self.screen_size = screen_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        # 将标签归一化到 [0, 1] 范围
        label = torch.tensor([label[0] / self.screen_size[0], label[1] / self.screen_size[1]], dtype=torch.float32)

        return image, label


class GazeJsonlDataset(Dataset):

    def __init__(self, data_paths,screen_size, transform=None):
        self.data_paths = data_paths
        self.screen_size = screen_size
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        for path in self.data_paths:
            with open(path, 'r') as file:
                for line in file:
                    data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        image=cv2.imread(image_path)
        label=item['label']
        if self.transform:
            image = self.transform(image)
        # 将标签归一化到 [0, 1] 范围
        label = torch.tensor([label[0] / self.screen_size[0], label[1] / self.screen_size[1]], dtype=torch.float32)

        return image, label


