import gzip, glob, os
import torch
import torch.nn as nn
import numpy as np
import scipy, scipy.misc
import imageio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_file_name(file_dir):
    filename = [name for name in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, name))]
    return filename

def get_file_num(file_dir):
    file_num = len([name for name in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, name))])
    return file_num

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def imread(path):
    return scipy.misc.imread(path)

def save_single(pred_dir, gen_dir, current_file_num, batch_size, image_size):
    num = int(np.floor(np.sqrt(batch_size)))
    for filename in glob.glob(pred_dir+'/pred_*'):
        image = scipy.misc.imread(filename)
        for idx in range(batch_size):
            i = idx % num
            j = idx // num
            img = image[j*image_size:j*image_size+image_size, i*image_size:i*image_size+image_size, :]
            scipy.misc.imsave(gen_dir+'/'+str(current_file_num+idx)+'.png', img)

def save_images(images, size, image_path):
    return imsave_merge(images, size, image_path)

def imsave_merge(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def imshow(image):
    return scipy.misc.imshow(image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)
 
def plot(hist, result_dir):
    x = range(len(hist['train_accuracy']))

    y1 = hist['train_loss']
    y2 = hist['train_accuracy']
    y3 = hist['test_accuracy']

    plt.plot(x, y1, label='train loss')
    plt.plot(x, y2, label='train accuracy')
    plt.plot(x, y3, label='test accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(result_dir, 'loss_acc.png')
    plt.savefig(path)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def lr_decay(optimizer, base_lr, epoch):
    lr = base_lr * (0.1 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

