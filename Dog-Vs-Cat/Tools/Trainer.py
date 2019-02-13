#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.autograd import Variable
from IPython.core.debugger import set_trace
from PIL import Image
from sklearn.metrics import confusion_matrix
import time

cuda_available = torch.cuda.is_available()
store_every = 200
start_epoch = 0
best_acc = torch.FloatTensor([0])
step = 10


class Trainer():
    def __init__(self,  model, optimizer, criterion, train_loader, valid_loader, test_loader, hyperparameters):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.hyperparameters = hyperparameters
        self.lr= hyperparameters['lr0']
        self.test_prediction = []
        self.log = {}
    # Keep only a single checkpoint, the best over test accuracy.
    def save_checkpoint(self, state, best, file_path="./output/checkpoint.pth.tar"):
        if best:
            print('new checkpoint is saved!')
            torch.save(state, file_path)
        else:
            print ('no improvement!')

          
    def evaluate(self, dataset_loader, criterion):
        LOSSES = 0
        COUNTER = 0
        for batch in dataset_loader:
            self.optimizer.zero_grad()

            x, y = batch
            y = y.view(-1)
            if cuda_available:
                x = x.cuda()
                y = y.cuda()

            loss = criterion(self.model(x), y)
            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            COUNTER += n

        return LOSSES / float(COUNTER)
    
    def accuracy(self, proba, y):
        correct = torch.eq(proba.max(1)[1], y).sum().type(torch.FloatTensor)
        return correct / y.size(0)
    
#     def adjust_lr(self, optimizer, epoch, total_epochs):
#         lr = lr0 * (0.5 ** (epoch / float(total_epochs)))
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

    def adjust_learning_rate(self, epoch):
        if (epoch+1) % step == 0:
            self.lr *= self.hyperparameters['gamma']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
    
    # def predict_test_set(self):
    #     results = [[]]
    #     for batch_idx, (inputs, targets) in enumerate(self.test_loader):
    #         if cuda_available:
    #             inputs, targets = inputs.cuda(), targets.cuda()
    #         outputs = self.model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         results = np.append(results, predicted.cpu().numpy())
        
    #     results = np.int8(results)
    #     self.predict_test_set = results

    # def generate_submission(self):
    #     self.predict_test_set()
    #     df = pd.DataFrame({ 'id': range(1, len(self.test_prediction)+1),'label': self.test_prediction})
    #     df['label'].replace([0,1], ['Cat','Dog'], inplace=True)
    #     df[df.columns].to_csv('submisstion.csv',index=False)
    #     print('Done...')

    def confusion_matrix(self):
        y_pred = [[]]
        y_true = [[]]
        for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = np.append(y_pred, predicted.cpu().numpy())
            y_true = np.append(y_true, targets.cpu().numpy())

        y_pred = np.int8(y_pred)
        return confusion_matrix(y_true, y_pred)
        # return [y_pred, y_true]
        
    def train_model(self):
        start_time = time.time()
        best_acc = torch.FloatTensor([0])
        c = 0
        LOSSES = 0
        COUNTER = 0
        ITERATIONS = 0
        learning_curve_nll_train = list()
        learning_curve_nll_test = list()
        learning_curve_acc_train = list()
        learning_curve_acc_test = list()
        for e in range(self.hyperparameters['num_epochs']):    

            print("------ Epoch #", e+1, "------")
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                x, y = batch
                y = y.view(-1)
                if cuda_available:
                    x = x.cuda()
                    y = y.cuda()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()

                n = y.size(0)
                LOSSES += loss.sum().data.cpu().numpy() * n
                COUNTER += n
                ITERATIONS += 1
                if ITERATIONS%(store_every/5) == 0:
                    avg_loss = LOSSES / float(COUNTER)
                    LOSSES = 0
                    COUNTER = 0
                    print(" Iteration {}: TRAIN {:.4f}".format(
                        ITERATIONS, avg_loss))

                if ITERATIONS%(store_every) == 0:     

                    train_loss = self.evaluate(self.train_loader, self.criterion)
                    learning_curve_nll_train.append(train_loss)
                    valid_loss = self.evaluate(self.valid_loader, self.criterion)
                    learning_curve_nll_test.append(valid_loss)

                    train_acc = self.evaluate(self.train_loader, self.accuracy)
                    learning_curve_acc_train.append(train_acc)
                    valid_acc = self.evaluate(self.valid_loader, self.accuracy)
                    learning_curve_acc_test.append(valid_acc)

                    print(" [Loss] TRAIN {:.4f} / VALID {:.4f}".format(
                        train_loss, valid_loss))
                    print(" [ACC] TRAIN {:.4f} / VALID {:.4f}".format(
                        train_acc, valid_acc))
                    
                    acc = torch.FloatTensor([valid_acc])
                    is_best = (acc.numpy() > best_acc.numpy())
                    best_accuracy = torch.FloatTensor(max(acc.numpy(), best_acc.numpy()))

                    # Save checkpoint if is a new best
                    if(self.hyperparameters['save_checkpoint']):
                        self.save_checkpoint({
                            'epoch': start_epoch + e + 1,
                            'state_dict': self.model.state_dict(),
                            'best_accuracy': best_accuracy
                        }, is_best)  
            if(self.hyperparameters['adjust_lr']):
                self.adjust_learning_rate(e)
        self.log = {'learning_curve_nll_train': learning_curve_nll_train,
                    'learning_curve_nll_test': learning_curve_nll_test,
                    'learning_curve_acc_train': learning_curve_acc_train,
                    'learning_curve_acc_test': learning_curve_acc_test
                    }
        return [learning_curve_nll_train, learning_curve_nll_test, learning_curve_acc_train,learning_curve_acc_test]

def predict_test_set(model, test_loader):
    results = [[]]
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs.view(-1,3, 64, 64)
        inputs= inputs.type(torch.cuda.FloatTensor)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        results = np.append(results, predicted.cpu().numpy())
    results = np.int8(results)
    return results

def predict_test_set_5crop(model, test_loader):
    results = [[]]
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()
        bs, ncrops, c, h, w = inputs.size()
        result = model(inputs.view(-1, c, h, w)) # fuse batch size and ncrops
        result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
        _, result_avg = torch.max(result_avg, 1)
        results = np.append(results, result_avg.cpu().detach().numpy())
    results = np.int8(results)
    return results

def generate_submission(results):
    df = pd.DataFrame({ 'id': range(1, len(results)+1),
                    'label': results})
    df['label'].replace([0,1], ['Cat','Dog'], inplace=True)
    df[df.columns].to_csv('submisstion.csv',index=False)
    print('Done...')
