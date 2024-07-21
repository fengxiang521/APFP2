import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate


class ProtoNet(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def feature_forward(self, x1,x2,x3):
        out1 = self.avgpool(x1).view(x1.size(0),-1)
        out2 = self.avgpool(x2).view(x2.size(0),-1)
        out3 = self.avgpool(x3).view(x3.size(0),-1)
        return out1,out2,out3

    def set_forward(self, x, is_feature=False):
        z_support, z_query,y_support, y_query,x_support, x_query = self.parse_feature(x, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        if self.n_support>1:
            scores1, new_scores1, new_scores2 = self.setO(z_support,z_query)
            return new_scores1, scores1, new_scores2
        else:
            scores = self.euclidean_dist(z_query, z_proto)
            return scores,scores,scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores,scores2,scores3 = self.set_forward(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        topk_scores2, topk_labels2 = scores2.data.topk(1, 1, True, True)
        topk_ind2 = topk_labels2.cpu().numpy()
        top1_correct2 = np.sum(topk_ind2[:, 0] == y_label)
        topk_scores3, topk_labels3 = scores3.data.topk(1, 1, True, True)
        topk_ind3 = topk_labels3.cpu().numpy()
        top1_correct3 = np.sum(topk_ind3[:, 0] == y_label)
        loss1=self.loss_fn(scores3, y_query)

        return float(top1_correct),float(top1_correct2),float(top1_correct3), len(y_label), loss1, scores3

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

    def setO(self,supports,querys):
        T1 = supports.contiguous().unsqueeze(0).expand(self.n_way*self.n_query, -1, -1,-1)
        T2 = querys.contiguous().unsqueeze(1).expand(-1, self.n_support, -1)
        T2=T2.contiguous().unsqueeze(1).expand(-1, self.n_support,-1,-1)
        query=querys.contiguous().unsqueeze(1).expand(-1, 5, -1)
        z_proto = supports.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_proto=z_proto.contiguous().unsqueeze(0).expand(self.n_way*self.n_query, -1, -1)
        dist = torch.pow(T1 - T2, 2).sum(3)
        score = -dist
        temperature = self.self.params.tem
        softmax_output2=F.softmax(score*temperature,dim=2)

        new_scores2=torch.pow(torch.sum(T1*softmax_output2.unsqueeze(3),dim=2)-query,2).sum(2)

        return  -new_scores2,-new_scores2,-new_scores2
