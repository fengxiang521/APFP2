import math
from sqlite3 import paramstyle
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from .bdc_module import *


class BaselineTrain(nn.Module):
    def __init__(self, params, model_func, num_class):
        super(BaselineTrain, self).__init__()
        self.params = params
        self.feature = model_func()
        if params.method in ['stl_deepbdc', 'meta_deepbdc']:
            reduce_dim = params.reduce_dim
            self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
            if self.params.dataset=='cub':
                self.dcov1 = BDC(is_vec=True, input_dim=self.feature.feat_dim[2], dimension_reduction=reduce_dim)
                self.dcov2 = BDC(is_vec=True, input_dim=self.feature.feat_dim[1], dimension_reduction=reduce_dim)
                self.dcov3 = BDC(is_vec=True, input_dim=self.feature.feat_dim[0], dimension_reduction=reduce_dim)
            else:
                self.dcov1 = BDC(is_vec=True, input_dim=self.feature.feat_dim[0], dimension_reduction=reduce_dim)
                self.dcov2 = BDC(is_vec=True, input_dim=self.feature.feat_dim[1], dimension_reduction=reduce_dim)
                self.dcov3 = BDC(is_vec=True, input_dim=self.feature.feat_dim[2], dimension_reduction=reduce_dim)
            self.dropout = nn.Dropout(params.dropout_rate)

        elif params.method in ['protonet', 'good_embed']:
            if self.params.dataset != 'cub':
                self.feat_dim = self.feature.feat_dim[0]
            else:
                self.feat_dim = self.feature.feat_dim[2]
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        if params.method in ['stl_deepbdc', 'meta_deepbdc', 'protonet', 'good_embed']:
            if self.params.dataset != 'cub':
                self.classifier = nn.Linear(self.feat_dim[0], num_class)
            else:
                self.classifier = nn.Linear(self.feat_dim[0], num_class)
            self.classifier.bias.data.fill_(0)

        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()

    def feature_forward(self, x):
        out1,out2,out3 = self.feature.forward(x)
        if self.params.method in ['stl_deepbdc', 'meta_deepbdc']:
            out1 = self.dcov1(out1)
            out2 = self.dcov2(out2)
            out3 = self.dcov3(out3)
            out1 = self.dropout(out1)
            out2 = self.dropout(out2)
            out3 = self.dropout(out3)
        elif self.params.method in ['protonet', 'good_embed']:
            out1 = self.avgpool(out1).view(out1.size(0), -1)
            out2 = self.avgpool(out2).view(out2.size(0), -1)
            out3 = self.avgpool(out3).view(out3.size(0), -1)
        return out1,out2,out3

    def forward(self, x):
        x = Variable(x.cuda())
        out1,out2,out3 = self.feature_forward(x)
        if self.params.method == 'protonet':
            scores1 = self.classifier.forward(out1)
            scores2 = self.classifier.forward(out1)
            scores3 = self.classifier.forward(out1)
        else:
            scores1 = self.classifier.forward(out1)
            scores2 = self.classifier.forward(out2)
            scores3 = self.classifier.forward(out3)
        return scores1,scores2,scores3

    def forward_meta_val(self, x):
        x = Variable(x.cuda())        
        x = x.contiguous().view(self.params.val_n_way * (self.params.n_shot + self.params.n_query), *x.size()[2:])
        
        out1,out2,out3 = self.feature_forward(x)

        z_all = out1.view(self.params.val_n_way, self.params.n_shot + self.params.n_query, -1)
        z_support = z_all[:, :self.params.n_shot]
        z_query = z_all[:, self.params.n_shot:]
        z_proto = z_support.contiguous().view(self.params.val_n_way, self.params.n_shot, -1).mean(1)
        z_query = z_query.contiguous().view(self.params.val_n_way * self.params.n_query, -1)

        if self.params.method in ['meta_deepbdc']:
            scores = self.metric(z_query, z_proto)
        elif self.params.method in ['protonet']:
            scores = self.euclidean_dist(z_query, z_proto)
        return scores

    def forward_loss(self, x, y):
        scores1,scores2,scores3 = self.forward(x)
        y = Variable(y.cuda())
        return self.loss_fn(scores1, y),self.loss_fn(scores2, y),self.loss_fn(scores3, y),scores1,scores2,scores3

    def forward_meta_val_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.params.val_n_way), self.params.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.params.val_n_way), self.params.n_query)
        scores = self.forward_meta_val(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 200
        avg_loss1 = 0
        avg_loss2 = 0
        avg_loss3 = 0
        total_correct1 = 0
        total_correct2 = 0
        total_correct3 = 0

        iter_num = len(train_loader)
        total = len(train_loader) * self.params.batch_size

        for i, (x, y) in enumerate(train_loader):
            y = Variable(y.cuda())
            optimizer.zero_grad()
            loss1,loss2,loss3, output1,output2,output3 = self.forward_loss(x, y)
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred3 = output3.data.max(1)[1]
            total_correct1 += pred1.eq(y.data.view_as(pred1)).sum()
            total_correct2 += pred2.eq(y.data.view_as(pred2)).sum()
            total_correct3 += pred3.eq(y.data.view_as(pred3)).sum()
            loss1.backward()
            optimizer.step()

            avg_loss3 = avg_loss3 + loss3.item()
            avg_loss2 = avg_loss2 + loss2.item()
            avg_loss1 = avg_loss1 + loss1.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss1 / float(i + 1)))
        return avg_loss1 / iter_num, float(total_correct1) / total * 100,avg_loss2 / iter_num, float(total_correct2) / total * 100,avg_loss3 / iter_num, float(total_correct3) / total * 100

    def test_loop(self, val_loader):
        total_correct = 0
        avg_loss = 0.0
        total = len(val_loader) * self.params.batch_size
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                y = Variable(y.cuda())
                loss, output = self.forward_loss(x, y)
                avg_loss = avg_loss + loss.item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(y.data.view_as(pred)).sum()
        avg_loss /= len(val_loader)
        acc = float(total_correct) / total
        # print('Test Acc = %4.2f%%, loss is %.2f' % (acc * 100, avg_loss))
        return avg_loss, acc * 100

    def meta_test_loop(self, test_loader):
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                correct_this, count_this, loss, _ = self.forward_meta_val_loss(x)
                acc_all.append(correct_this / count_this * 100)
                avg_loss = avg_loss + loss.item()
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return avg_loss / iter_num, acc_mean

    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        if self.params.n_shot > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            score = (x * y).sum(2)
        return score
    
    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score


class MetaTemplate(nn.Module):
    def __init__(self, params, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = params.n_query  # (change depends on input)
        self.feature = model_func()
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.params = params

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    @abstractmethod
    def feature_forward(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)[0]
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])

            #x = self.feature.forward(x) # ==>resnet.forward 117
            x1,x2,x3=self.feature.forward(x)
            z_all,y_all,x_all=self.feature_forward(x1,x2,x3)
            #z_all = self.feature_forward(x1) # ==>meta_deepbdc.feature_forward 20
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
            y_all = y_all.view(self.n_way, self.n_support + self.n_query, -1)
            x_all = x_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        y_support = y_all[:, :self.n_support]
        x_support = x_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        y_query = y_all[:, self.n_support:]
        x_query = x_all[:, self.n_support:]

        return z_support, z_query,y_support, y_query,x_support, x_query

    def get_feature_X(self ,x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])

            # x = self.feature.forward(x) # ==>resnet.forward 117
            x1, x2, x3 = self.feature.forward(x)
            #z_all, y_all, x_all = self.feature_forward(x1, x2, x3)
        return x1, x2, x3

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)



    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 100
        avg_loss = 0
        acc_all = []
        iter_num = len(train_loader)
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            correct_this,correct_this2,correct_this3, count_this, loss, _ = self.set_forward_loss(x)#==》meta_deepbdc
            acc_all.append(correct_this / count_this * 100)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        return avg_loss / iter_num, acc_mean

    def test_loop(self, test_loader, record=None):
        acc_all = []
        acc_all2 = []
        acc_all3 = []
        avg_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
                correct_this,correct_this2,correct_this3, count_this, loss, _ = self.set_forward_loss(x)
                acc_all.append(correct_this / count_this * 100)
                acc_all2.append(correct_this2 / count_this * 100)
                acc_all3.append(correct_this3 / count_this * 100)
                avg_loss = avg_loss + loss.item()
        acc_all = np.asarray(acc_all)
        acc_all2 = np.asarray(acc_all2)
        acc_all3 = np.asarray(acc_all3)
        acc_mean = np.mean(acc_all)
        acc_mean2 = np.mean(acc_all2)
        acc_mean3 = np.mean(acc_all3)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean2, 1.96 * acc_std / np.sqrt(iter_num)))
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean3, 1.96 * acc_std / np.sqrt(iter_num)))

        return avg_loss / iter_num, acc_mean