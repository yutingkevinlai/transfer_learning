from __future__ import division
from utils import bcolors
import os, utils, utilsLoadData, time, pickle, cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

class lenet(nn.Module):
    def __init__(self, dataset='wood', img_size=128):
        super(lenet, self).__init__()
        self.input_height = img_size
        self.input_width = img_size
        self.input_dim = 3
        if dataset == 'wood':
            self.output_dim = 5
        elif dataset == 'DAGM_8':
            self.output_dim = 2
        elif dataset == 'DAGM_10':
            self.output_dim = 2
        elif dataset == 'middle_white':
            self.output_dim = 2
        elif dataset == 'solar':
            self.output_dim = 2
        elif dataset == 'flower_chip':
            self.output_dim = 2
        elif dataset == 'gas_leak_dirt':
            self.output_dim = 2
        elif dataset == 'intra_chip_diff':
            self.output_dim = 2
        else:
            raise Exception("[!] No dataset named %s" %dataset)
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 6, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*(self.input_height//4)*(self.input_width//4), 120),
            nn.ReLU(True),
            nn.Linear(120, self.output_dim),
            nn.Softmax(),
        )

        utils.initialize_weights(self.conv)
        utils.initialize_weights(self.classifier)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 16*(self.input_height//4)*(self.input_width//4))
        x = self.classifier(x)
        return x

class vgg19(nn.Module):
    def __init__(self, dataset='wood', img_size=128):
        super(vgg19, self).__init__()
        self.input_height = img_size
        self.input_width = img_size
        self.input_dim = 3
        if dataset == 'wood':
            self.output_dim = 5
        elif dataset == 'DAGM_8':
            self.output_dim = 2
        elif dataset == 'DAGM_10':
            self.output_dim = 2
        elif dataset == 'middle_white':
            self.output_dim = 2
        elif dataset == 'solar':
            self.output_dim = 2
        elif dataset == 'flower_chip':
            self.output_dim = 2
        elif dataset == 'gas_leak_dirt':
            self.output_dim = 2
        elif dataset == 'intra_chip_diff':
            self.output_dim = 2
        else:
            raise Exception("[!] No dataset named %s" %dataset)

        self.vgg = models.vgg19_bn(pretrained=True)
        #for param in self.vgg.parameters():
        #    param.requires_grad = False
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512*(self.input_height//(2**5))*(self.input_width//(2**5)), 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.output_dim),
            nn.Softmax(),
        )
        #self.vgg.classifier._modules['6'] = nn.Linear(4096, 5)
        #utils.initialize_weights(self.vgg.classifier._modules['6'])
        utils.initialize_weights(self.vgg.classifier)

    def forward(self, x):
        x = self.vgg(x)
        return x

class transfer(object):
    def __init__(self, args):
        self.date = args.date
        self.dataset = args.dataset
        self.img_size = args.img_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.decay = args.decay
        self.result_dir = os.path.join(args.result_folder, args.dataset, args.network, args.date)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        if args.network == 'vgg':
            self.network = vgg19(self.dataset, self.img_size)
        else:
            self.network = lenet(self.dataset, self.img_size)
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr)

        self.network.cuda()
        self.BCE_loss = nn.BCELoss().cuda()

        print("---------- Network architecture ----------")
        utils.print_network(self.network)
        print("------------------------------------------")

    def train(self):
        self.early_stop = False
        self.train_hist = {}
        self.train_hist['train_loss'] = []
        self.train_hist['train_accuracy'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.record_file = os.path.join(self.result_dir,"records_"+str(self.img_size)+".txt")

        # read data for training stage
        if self.dataset == 'wood':
            self.train_data_loader = utilsLoadData.load_all(
                data_dir='../data/wood', 
                batch_size=self.batch_size, 
                img_size=self.img_size)
        elif self.dataset == 'DAGM_8':
            self.train_data_loader = utilsLoadData.load_all(
                data_dir='../data/DAGM_8',
                batch_size=self.batch_size,
                img_size=self.img_size)
        elif self.dataset == 'DAGM_10':
            self.train_data_loader = utilsLoadData.load_all(
                data_dir='../data/DAGM_10',
                batch_size=self.batch_size,
                img_size=self.img_size)
        elif self.dataset == 'middle_white':
            self.train_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../data/middle_white/OK_train_128', '../data/middle_white/NG_train_128')
        elif self.dataset == 'flower_chip':
            # because flower_chip flaw is polycrystaline, so OK samples are different than other monocrystaline
            self.train_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/flower_chip/OK/train', '../generative_models/pytorch/data/flower_chip/NG/train')
        elif self.dataset == 'gas_leak_dirt':
            self.train_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/solar_128', '../generative_models/pytorch/data/gas_leak_dirt/train')
        elif self.dataset == 'intra_chip_diff':
            self.train_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/solar_128', '../generative_models/pytorch/data/intra_chip_diff')
        else:
            raise Exception("[!] No dataset named %s" %self.dataset)

        # start training
        self.network.train()
        print(bcolors.OKBLUE+"Training start! total: %d data" %self.train_data_loader.dataset.__len__()+bcolors.ENDC)
        with open(self.record_file, 'w') as f:
            f.write("learning rate: %f\n" %self.lr)
        start_time = time.time()
        for epoch in range(self.epoch):
            # initialize some variables to calculate confusion matrix
            neg = torch.zeros(self.batch_size, 1).type(torch.LongTensor).cuda()
            pos = torch.ones(self.batch_size, 1).type(torch.LongTensor).cuda()
            corrects = 0
            self.train_tp, self.train_fn, self.train_fp, self.train_tn = 0, 0, 0, 0
            utils.lr_decay(self.optimizer, (epoch+1), self.decay, mode='lambda')

            # start epoch
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.train_data_loader):
                if iter == (self.train_data_loader.dataset.__len__()//self.batch_size):
                    neg = torch.zeros(len(x_[:]), 1).type(torch.LongTensor).cuda()
                    pos = torch.ones(len(x_[:]), 1).type(torch.LongTensor).cuda()
                
                x_, y_ = Variable(x_.cuda()), Variable(y_.type(torch.FloatTensor).cuda())

                # update betwork
                self.optimizer.zero_grad()
 
                outputs = self.network(x_)
                _, preds = torch.max(outputs.data, 1, keepdim=True) # find the max pos
                _, gts = torch.max(y_.data, 1, keepdim=True) # find the max pos
                loss = self.BCE_loss(outputs, y_)

                loss.backward()
                self.optimizer.step()

                # statistics
                corrects += torch.sum(preds == gts)
                self.train_tp += torch.sum(torch.gt((preds==neg),(gts==pos)))
                self.train_fn += torch.sum(torch.gt(preds,gts))
                self.train_tn += torch.sum(torch.lt((preds==neg),(gts==pos)))
                self.train_fp += torch.sum(torch.lt(preds,gts))

                if ((iter+1)%100 == 0):
                    print("Epoch: [%2d] [%4d/%4d] train loss: %.8f" %((epoch + 1), (iter + 1), self.train_data_loader.dataset.__len__()//self.batch_size, loss.data[0]))

            self.train_accuracy = corrects / self.train_data_loader.dataset.__len__()
            self.train_hist['train_loss'].append(loss.data[0])
            self.train_hist['train_accuracy'].append(self.train_accuracy)
            print(bcolors.OKGREEN+bcolors.BOLD+"train accuracy: %.4f" %(self.train_accuracy))
            print("--------------------------------------")
            print("|          |  positive  |  negative  |")
            print("| positive |   %7d  |   %7d  |" %(self.train_tp, self.train_fp))
            print("| negative |   %7d  |   %7d  |" %(self.train_fn, self.train_tn))
            print("--------------------------------------"+bcolors.ENDC)
            with open(self.record_file, 'a') as f:
                f.write("Epoch: %d\n" %(epoch+1))
                f.write("Training Accuracy: %.4f\n" %(self.train_accuracy))
                f.write("--------------------------------------\n")
                f.write("|          |  positive  |  negative  |\n")
                f.write("| positive |   %7d  |   %7d  |\n" %(self.train_tp, self.train_fp))
                f.write("| negative |   %7d  |   %7d  |\n" %(self.train_fn, self.train_tn))
                f.write("--------------------------------------\n")

            # compute testing accuracy and print testing information
            if self.dataset == 'wod':
                self.predict_test()

                with open(self.record_file, 'a') as f:
                    f.write("Testing Accuracy: %.4f\n" %(self.test_accuracy))
                    f.write("--------------------------------------\n")
                    f.write("|          |  positive  |  negative  |\n")
                    f.write("| positive |   %7d  |   %7d  |\n" %(self.test_tp, self.test_fp))
                    f.write("| negative |   %7d  |   %7d  |\n" %(self.test_fn, self.test_tn))
                    f.write("--------------------------------------\n")

            self.train_hist['per_epoch_time'].append(time.time()-epoch_start_time)
            # early stopping
            if not self.dataset == 'wood':
                if (self.train_accuracy>0.996) and (self.test_accuracy>0.996):
                    self.save()
                    self.early_stop = True
                    print("[!] Early stopping!")
                    with open(self.record_file, 'a') as f:
                        f.write("[!] Early stopping")
                    break
            
        self.train_hist['total_time'].append(time.time()-start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" %(np.mean(self.train_hist['per_epoch_time']), self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... saving training results")

        # save model
        utils.plot(self.train_hist, self.result_dir)
        if not self.early_stop:
            self.save()

    def save(self):
        torch.save(self.network.state_dict(), os.path.join(self.result_dir, 'model_'+str(self.img_size)+'.pkl'))

    def load(self):
        self.network.load_state_dict(torch.load(os.path.join(self.result_dir, 'model_'+str(self.img_size)+'.pkl')))

    def predict_test(self):
        self.network.eval()
        if self.dataset == 'middle_white':
            self.test_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data//middle_white/OK_train_128', '../generative_models/pytorch/data/middle_white/NG_test_128')
        elif self.dataset == 'flower_chip':
            self.test_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/flower_chip/OK/test', '../generative_models/pytorch/data/flower_chip/NG/test')
        elif self.dataset == 'gas_leak_dirt':
            self.test_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/solar_128', '../generative_models/pytorch/data/gas_leak_dirt/test')
        corrects = 0
        self.test_tp, self.test_tn, self.test_fp, self.test_fn = 0, 0, 0, 0
        neg = torch.zeros(self.batch_size, 1).type(torch.LongTensor).cuda()
        pos = torch.ones(self.batch_size, 1).type(torch.LongTensor).cuda()
        for iter, (x_, y_) in enumerate(self.test_data_loader):
            if iter == (self.test_data_loader.dataset.__len__()//self.batch_size):
                neg = torch.zeros(len(x_[:]), 1).type(torch.LongTensor).cuda()
                pos = torch.ones(len(x_[:]), 1).type(torch.LongTensor).cuda()

            x_, y_ = Variable(x_.cuda()), Variable(y_.type(torch.FloatTensor).cuda())
            outputs = self.network(x_)

            _, preds = torch.max(outputs.data, 1, keepdim=True)
            _, gts = torch.max(y_.data, 1, keepdim=True)

            # statistics
            corrects += torch.sum(preds == gts)
            self.test_tp += torch.sum(torch.gt((preds==neg),(gts==pos)))
            self.test_fn += torch.sum(torch.gt(preds,gts))
            self.test_tn += torch.sum(torch.lt((preds==neg),(gts==pos)))
            self.test_fp += torch.sum(torch.lt(preds,gts))

        self.test_accuracy = corrects / self.test_data_loader.dataset.__len__()

        print(bcolors.OKGREEN+bcolors.BOLD+"\ntest accuracy: %.4f" %(self.test_accuracy))
        print("--------------------------------------")
        print("|          |  positive  |  negative  |")
        print("| positive |   %7d  |   %7d  |" %(self.test_tp, self.test_fp))
        print("| negative |   %7d  |   %7d  |" %(self.test_fn, self.test_tn))
        print("--------------------------------------"+bcolors.ENDC)

        print("\n"+bcolors.OKGREEN+bcolors.BOLD+"Precision: %.8f" %(self.test_tp/(self.test_tp+self.test_fp)))
        print(bcolors.OKGREEN+bcolors.BOLD+"False Positive Rate: %.8f" %(self.test_fp/(self.test_fp+self.test_tn))+"\n"+bcolors.ENDC)

    def predict_gen(self):
        # Make prediction on generated data
        self.load()

        self.network.eval()
        self.predict_data_loader = utilsLoadData.load_one(self.batch_size, self.img_size, data_dir='../generative_models/pytorch/0101_DAGAN/solar_128/DAGAN/generated')
        corrects = 0
        for iter, (x_, y_) in enumerate(self.predict_data_loader):
            x_, y_ = Variable(x_.cuda()), Variable(y_.type(torch.FloatTensor).cuda())
            outputs = self.network(x_)

            _, preds = torch.max(outputs.data, 1, keepdim=True)
            _, gts = torch.max(y_.data, 1, keepdim=True)
            corrects += torch.sum(preds == gts)

        accuracy = corrects / self.predict_data_loader.dataset.__len__()
        print("Accuracy of generated data: %.4f" %accuracy)
