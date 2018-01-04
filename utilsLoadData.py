import glob, os, random
import torch.nn as nn
import numpy as np
import scipy, scipy.misc
import imageio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def load_wood_old(data_dir, batch_size, img_size=128):
    transform = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()])
    dataset = folder_five(data_dir, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

'''custom dataset compatible with rebuilt DataLoader'''
def load_celebA(data_dir, batch_size, img_size=64):
    transform = transforms.Compose([transforms.CenterCrop(160), transforms.Scale(32), transforms.ToTensor()])
    dataset = celeba_folder(data_dir, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader

class celeba_folder(Dataset):
    '''celebA dataset'''
    def __init__(self, data_dir, transform=None):
        self.image_paths = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = np.zeros((10), dtype=np.float)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)    

def load_wood(data_dir, batch_size, img_size=100, convert='RGB'):
    transform = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()]) # do not need to normalize
    dataset = wood_folder(data_dir, transform, convert)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader

class wood_folder(Dataset):    
    '''wood texture dataset'''
    def __init__(self, data_dir, transform=None, convert='RGB'):
        '''initialize image paths and preprocessing module'''
        # collect image paths using map
        self.image_paths = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
        self.transform = transform
        self.convert = convert

    def __getitem__(self, index):
        '''reads an image from a file and preprocesses it and returns'''
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert(self.convert)
        label = np.zeros((10), dtype=np.float)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
 
    def __len__(self):
        '''return the total number of image files'''
        return len(self.image_paths)

def load_one(batch_size, img_size, data_dir):
    transform = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()])
    dataset = folder_one(transform, data_dir)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

class folder_one(Dataset):
    def __init__(self, transform, data_dir):
        self.image_paths = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        label = np.array([1,0], dtype=np.float)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

def sample(_list, _num):
    rand_num = random.sample(range(0, len(_list)), _num)
    return [_list[rand_num[i]] for i in range(len(rand_num))]

def load_two(batch_size, img_size, *args):
    transform = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()])
    dataset = folder_two(transform, *args)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

class folder_two(Dataset):
    def __init__(self, transform, *args):
        image_paths = {}
        lengths = []
        self.arg_length = len(args)
        self.image_paths = []
        for iter, arg in enumerate(args):
            image_paths[iter] = list(map(lambda x: os.path.join(arg, x), os.listdir(arg)))
            lengths.append(len(image_paths[iter]))
        self.least = min(lengths)
        for i in range(len(image_paths)):
            self.image_paths += sample(image_paths[i], self.least)
            print(len(self.image_paths))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if index < self.least:
            label = np.array([1,0], dtype=np.float)
        elif index < self.least*2:
            label = np.array([0,1], dtype=np.float)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

class folder_five(Dataset):
    def __init__(self, transform, *args):
        image_paths = {}
        lengths = []
        self.arg_length = len(args)
        self.image_paths = []
        for iter, arg in enumerate(args):
            image_paths[iter] = list(map(lambda x: os.path.join(arg, x), os.listdir(arg)))
            lengths.append(len(image_paths[iter]))
        self.least = min(lengths)
        for i in range(len(image_paths)):
            self.image_paths += sample(image_paths[i], self.least)
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if index < self.least:
            label = np.array([1,0,0,0,0], dtype=np.float)
        elif index < self.least*2:
            label = np.array([0,1,0,0,0], dtype=np.float)
        elif index < self.least*3:
            label = np.array([0,0,1,0,0], dtype=np.float)
        elif index < self.least*4:
            label = np.array([0,0,0,1,0], dtype=np.float)
        elif index < self.least*5:
            label = np.array([0,0,0,0,1], dtype=np.float)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

def load_solar(data_dir, batch_size, img_size=100):
    transform = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()])
    dataset = solar_folder(data_dir, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader
    
class solar_folder(Dataset):
    ''' solar panel '''
    def __init__(self, data_dir, transform=None):
        self.image_paths = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        #image = Image.open(image_path).convert('L')

        label = np.ones((10), dtype=np.float)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

