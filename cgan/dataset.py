import torch
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils import data
from torchvision import transforms
import os
import cv2
from random import random

def read_rgb(src:str):
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class Dataset(data.Dataset):
    def __init__(self,train:bool=True) -> None:
        super().__init__()
        path = './data/maps/train/' if train else './data/maps/val/'
        
        self.data = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and '.jpg' in f]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,512)),
        ])

        self.normalize = transforms.Normalize(.5,.5)
        
    def __len__(self):
        return len(self.data)
    

    def random_crop(self, original, transformed):
        stacked_img = torch.stack((original, transformed), 0)
        img_crop = transforms.RandomCrop((stacked_img.shape[-2], stacked_img.shape[-1]))(stacked_img)
        return img_crop[0], img_crop[1]
    


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()

        img = read_rgb(self.data[index])
        img = self.transform(img)
        
        w = img.shape[-1]//2
        
        original_img = img[ :, :, :w]
        trasformed_img = img[ :, :, w:]

        original_img, trasformed_img = self.random_crop(original=original_img, transformed=trasformed_img)
        
        if random() > 0.5:
            original_img = torch.fliplr(original_img)
            trasformed_img = torch.fliplr(trasformed_img)
        
        original_img = self.normalize(original_img)
        trasformed_img = self.normalize(trasformed_img)

        return original_img, trasformed_img








def get_dataset(train:bool=True):
    if not os.path.isdir('./data/maps'):
        dataset = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz'
        dataset = download_and_extract_archive(dataset, './data/')
    
    dataset = Dataset(train=train)
    
    return dataset
    
