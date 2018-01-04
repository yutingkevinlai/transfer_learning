from __future__ import division
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

class vgg19(nn.Module):
    def __init__(self, dataset='wood_old', img_size=128):
        super(vgg19, self).__init__()
        self.input_height = img_size
        self.input_width = img_size
        self.input_dim = 3
        if dataset == 'wood_old':
            self.output_dim = 5
        elif dataset == 'middle_white':
            self.output_dim = 2
        elif dataset == 'solar_ok_ori_128':
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
        self.dataset = args.dataset
        self.img_size = args.img_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.result_dir = os.path.join(args.result_folder, args.dataset)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.network = vgg19(self.dataset, self.img_size)
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr)

        self.network.cuda()
        self.BCE_loss = nn.BCELoss().cuda()

        print("---------- Network architecture ----------")
        utils.print_network(self.network)
        print("------------------------------------------")

        if self.dataset == 'wood_old':
            self.train_data_loader = utilsLoadData.load_wood_old(data_dir='../generative_models/pytorch/data/wood_old', batch_size=self.batch_size, img_size=self.img_size)
        elif self.dataset == 'middle_white':
            self.train_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/solar_ok_ori_128', '../generative_models/pytorch/data/middle_white/NG_train_128')
        elif self.dataset == 'flower_chip':
            self.train_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/solar_ok_ori_128', '../generative_models/pytorch/data/flower_chip')
        elif self.dataset == 'gas_leak_dirt':
            self.train_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/solar_ok_ori_128', '../generative_models/pytorch/data/gas_leak_dirt/train')
        elif self.dataset == 'intra_chip_diff':
            self.train_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/solar_ok_ori_128', '../generative_models/pytorch/data/intra_chip_diff')
        else:
            raise Exception("[!] No dataset named %s" %self.dataset)

    def train(self):
        self.train_hist = {}
        self.train_hist['train_loss'] = []
        self.train_hist['train_accuracy'] = []
        self.train_hist['test_accuracy'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.record_file = os.path.join(self.result_dir,"records_"+str(self.img_size)+".txt")

        self.network.train()
        print("Training start! with %d data" %self.train_data_loader.dataset.__len__())
        start_time = time.time()
        for epoch in range(self.epoch):
            # initialize some variables to calculate confusion matrix
            neg = torch.zeros(self.batch_size, 1).type(torch.LongTensor).cuda()
            pos = torch.ones(self.batch_size, 1).type(torch.LongTensor).cuda()
            corrects = 0
            true_positive, false_negative, false_positive, true_negative = 0, 0, 0, 0
            utils.lr_decay(self.optimizer, self.lr, (epoch+1))

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
                _, preds = torch.max(outputs.data, 1, keepdim=True)
                _, gts = torch.max(y_.data, 1, keepdim=True)
                loss = self.BCE_loss(outputs, y_)

                loss.backward()
                self.optimizer.step()

                # statistics
                corrects += torch.sum(preds == gts)
                true_positive += torch.sum(torch.gt((preds==neg),(gts==pos)))
                false_positive += torch.sum(torch.gt(preds,gts))
                true_negative += torch.sum(torch.lt((preds==neg),(gts==pos)))
                false_negative += torch.sum(torch.lt(preds,gts))

                if ((iter+1)%100 == 0):
                    print("Epoch: [%2d] [%4d/%4d] train loss: %.8f" %((epoch + 1), (iter + 1), self.train_data_loader.dataset.__len__()//self.batch_size, loss.data[0]))

            train_accuracy = corrects / self.train_data_loader.dataset.__len__()
            self.train_hist['train_loss'].append(loss.data[0])
            self.train_hist['train_accuracy'].append(train_accuracy)
            print("train accuracy: %.4f" %(train_accuracy))
            print("----------------------------------")
            print("|          |  positive  |  negative  |")
            print("| positive |   %7d  |   %7d  |" %(true_positive, false_negative))
            print("| negative |   %7d  |   %7d  |" %(false_positive, true_negative))
            print("-----------------------------------")
            with open(self.record_file, 'a') as f:
                f.write("Epoch: %d\n" %(epoch+1))
                f.write("Training Accuracy: %.4f\n" %(train_accuracy))
                f.write("--------------------------------------\n")
                f.write("|          |  positive  |  negative  |\n")
                f.write("| positive |   %7d  |   %7d  |\n" %(true_positive, false_negative))
                f.write("| negative |   %7d  |   %7d  |\n" %(false_positive, true_negative))
                f.write("--------------------------------------\n")

            # compute testing accuracy and print testing information
            test_accuracy = self.predict_test()

            self.train_hist['per_epoch_time'].append(time.time()-epoch_start_time)
            # early stopping
            if (train_accuracy>0.996) and (test_accuracy>0.996):
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
            self.test_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/solar_ok_ori_128', '../generative_models/pytorch/data/middle_white/NG_test_128')
        elif self.dataset == 'gas_leak_dirt':
            self.test_data_loader = utilsLoadData.load_two(self.batch_size, self.img_size, '../generative_models/pytorch/data/solar_ok_ori_128', '../generative_models/pytorch/data/gas_leak_dirt/test')
        corrects = 0
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
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
            true_positive += torch.sum(torch.gt((preds==neg),(gts==pos)))
            false_positive += torch.sum(torch.gt(preds,gts))
            true_negative += torch.sum(torch.lt((preds==neg),(gts==pos)))
            false_negative += torch.sum(torch.lt(preds,gts))

        test_accuracy = corrects / self.test_data_loader.dataset.__len__()
        self.train_hist['test_accuracy'].append(test_accuracy)

        print("test accuracy: %.4f" %(test_accuracy))
        print("----------------------------------")
        print("|          |  positive  |  negative  |")
        print("| positive |   %7d  |   %7d  |" %(true_positive, false_negative))
        print("| negative |   %7d  |   %7d  |" %(false_positive, true_negative))
        print("-----------------------------------")

        with open(self.record_file, 'a') as f:
            f.write("Testing Accuracy: %.4f\n" %(test_accuracy))
            f.write("--------------------------------------\n")
            f.write("|          |  positive  |  negative  |\n")
            f.write("| positive |   %7d  |   %7d  |\n" %(true_positive, false_negative))
            f.write("| negative |   %7d  |   %7d  |\n" %(false_positive, true_negative))
            f.write("--------------------------------------\n")

        return test_accuracy

    def predict(self):
        # Make prediction on generated data
        self.load()

        self.network.eval()
        self.predict_data_loader = utilsLoadData.load_one(self.batch_size, self.img_size, data_dir='../generative_models/pytorch/0101_DAGAN/solar_ok_ori_128/DAGAN/generated')
        corrects = 0
        for iter, (x_, y_) in enumerate(self.predict_data_loader):
            x_, y_ = Variable(x_.cuda()), Variable(y_.type(torch.FloatTensor).cuda())
            outputs = self.network(x_)

            _, preds = torch.max(outputs.data, 1, keepdim=True)
            _, gts = torch.max(y_.data, 1, keepdim=True)
            corrects += torch.sum(preds == gts)

        accuracy = corrects / self.predict_data_loader.dataset.__len__()
        print("Accuracy of generated data: %.4f" %accuracy)
