import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

class Client(object):
    def __init__(self, name, model, classifier, local_bs, local_ep, lr, momentum, weight_decay, device, 
                 train_dl_local = None, test_dl_local = None):
        
        self.name = name 
        self.net = model
        self.clf = classifier
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr 
        self.momentum = momentum 
        self.weight_decay = weight_decay
        self.device = device 
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0 
        self.count = 0 
        self.save_best = True
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr/10, momentum=self.momentum, weight_decay=self.weight_decay) 
        self.clf_optimizer = torch.optim.SGD(self.clf.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        
    def train(self, clf_list, sims, inter_coeff, is_print = False):
        self.net.to(self.device)
        self.clf.to(self.device)
        self.net.train()
        self.clf.train()

        for clf in clf_list:
            clf.to(self.device)
        
        
        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for (images, images_aug, labels) in tqdm(self.ldr_train, desc = self.name):
                images, images_aug, labels = images.to(self.device), images_aug.to(self.device), labels.to(self.device)

                feature = self.net(images)
                output = self.clf(feature)

                feature_aug = self.net(images_aug)
                output_aug = self.clf(feature_aug)

                src_loss1 = self.loss_func(output_aug, labels)
                src_loss2 = self.loss_func(output, labels)
                
                task_loss_s = 0.5 * src_loss1 + 0.5 * src_loss2

                inter_loss = 0

                for i in range(len(clf_list)):
                    feature = self.net(images)
                    output_inter = clf_list[i](feature)

                    if i == 0:
                        inter_loss = sims[i] * self.loss_func(output_inter, labels)
                    
                    else:
                        inter_loss += sims[i] * self.loss_func(output_inter, labels)
                
                inter_loss /= len(clf_list)

                loss = task_loss_s + inter_coeff * inter_loss

                self.optimizer.zero_grad()
                self.clf_optimizer.zero_grad()
                loss.backward() 
                        
                self.optimizer.step()
                self.clf_optimizer.step()

                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
#         if self.save_best: 
#             _, acc = self.eval_test()
#             if acc > self.acc_best:
#                 self.acc_best = acc 

        self.net.cpu()
        self.clf.cpu()

        for clf in clf_list:
            clf.cpu()
        
        return sum(epoch_loss) / len(epoch_loss)
    
    def get_state_dict(self, mode):
        if mode == 'clf':
            return self.clf.state_dict()
            
        return self.net.state_dict()
    def get_best_acc(self):
        return self.acc_best
    def get_count(self):
        return self.count
    def get_net(self):
        return self.net
    def get_clf(self):
        return self.clf
    def set_state_dict(self, net_state_dict, clf_state_dict):
        self.net.load_state_dict(net_state_dict)
        self.clf.load_state_dict(clf_state_dict)

    def eval_test(self):
        self.net.to(self.device)
        self.clf.to(self.device)
        self.net.eval()
        self.clf.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                feature = self.net(data)
                output = self.clf(feature)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy
    
    def eval_test_glob(self, net_glob, clf_glob):
        net_glob.to(self.device)
        clf_glob.to(self.device)
        net_glob.eval()
        clf_glob.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = clf_glob(net_glob(data))
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy
    
    def eval_train(self):
        self.net.to(self.device)
        self.clf.to(self.device)
        self.net.eval()
        self.clf.eval()

        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data1, _, target in self.ldr_train:
                data1, target = data1.to(self.device), target.to(self.device)
                output = self.clf(self.net(data1))

                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy