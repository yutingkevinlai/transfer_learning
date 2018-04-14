import glob, os, csv, cv2
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

def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/raw/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/raw/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/raw/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/raw/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

'''custom dataset compatible with rebuilt DataLoader'''
def load(data_dir, batch_size, img_size=64, convert='RGB'):
    ''' load data in general'''
    transform = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()]) # do not need to normalize
    dataset = dataFolder(data_dir, transform, convert)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader

class dataFolder(Dataset):    
    ''' folder for general dataset '''
    def __init__(self, data_dir, transform=None, convert='RGB'):
        '''initialize image paths and preprocessing module'''
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

def load_all(data_dir, batch_size, img_size=64, convert='RGB'):
    transform = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()])
    dataset = dataFolderAll(data_dir, transform, convert)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader
    
class dataFolderAll(Dataset):
    def __init__(self, data_dir, transform=None, convert='RGB'):
        self.data_dir = data_dir
        self.subfolders = list(map(lambda x: os.path.join(self.data_dir,x), os.listdir(self.data_dir)))
        print(self.subfolders)
        self.image_paths = []
        self.labels = []
        for i in range(len(self.subfolders)):
            imgs = list(map(lambda x: os.path.join(self.subfolders[i],x), os.listdir(self.subfolders[i])))
            label = np.zeros(len(self.subfolders))
            print(label)
            for j in range(len(imgs)):
                self.image_paths.append(imgs[j])
                label[i] = 1
                self.labels.append(label)
        print(len(self.image_paths))
        print(len(self.labels))
        self.transform = transform
        self.convert = convert

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert(self.convert)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]
        
    def __len__(self):
        return len(self.image_paths)
       

def load_solar_all(data_ok_dir, data_ng_1_dir, data_ng_2_dir, batch_size):
    transform = transforms.Compose([transforms.Scale(100), transforms.ToTensor()])
    dataset = solar_all_folder(data_ok_dir, data_ng_1_dir, data_ng_2_dir, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader
 
class solar_all_folder(Dataset):
    """for CGAN training"""
    def __init__(self, data_ok_dir, data_ng_1_dir, data_ng_2_dir, transform=None):
        self.image_ok_paths_all = list(map(lambda x: os.path.join(data_ok_dir,x), os.listdir(data_ok_dir)))

        self.image_ng_paths_1 = list(map(lambda x: os.path.join(data_ng_1_dir,x), os.listdir(data_ng_1_dir)))
        self.image_ng_paths_2 = list(map(lambda x: os.path.join(data_ng_2_dir,x), os.listdir(data_ng_2_dir)))
        self.image_ng_paths = self.image_ng_paths_1 + self.image_ng_paths_2

        # because there are fewer samples in defect images, so we need to sample from the qualified images so that they are of same size
        samples = np.random.randint(len(self.image_ok_paths_all), size=len(self.image_ng_paths))
        self.image_ok_paths = [self.image_ok_paths_all[i] for i in samples]
        #print(len(self.image_ok_paths))
    
        self.image_paths = self.image_ok_paths + self.image_ng_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if index < len(self.image_paths)//2:
            label = np.ones((10), dtype=np.float)
        else:
            label = np.zeros((10), dtype=np.float)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        #print(len(self.image_paths))
        return len(self.image_paths)

def load_multi(data_dir, batch_size, img_size=64, convert='RGB'):
    transform = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()])
    dataset = data_folder_multi(data_dir, transform, convert)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader
      
class data_folder_multi(Dataset):
    def __init__(self, data_dir, transform=None, convert='RGB'):
        self.image_paths = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
        self.image_paths = self.check_image_folder()
        print(len(self.image_paths))
        self.transform = transform
        self.convert = convert
    def check_image_folder(self):
        length = len(self.image_paths)
        if length < 10:
            paths = []
            for i in range(length):
                folder = self.image_paths[i]
                tmp_paths = list(map(lambda x: os.path.join(folder,x), os.listdir(folder)))
                paths = paths + tmp_paths
            return paths
        else:
            return 1
                         
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert(self.convert)
        label = np.zeros((10), dtype=np.float)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.image_paths)

