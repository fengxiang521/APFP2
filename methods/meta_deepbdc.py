import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate
from .bdc_module import BDC
###############
import numpy as np
from scipy.stats import kendalltau
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance
from sklearn.metrics.pairwise import rbf_kernel

class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        
        reduce_dim = params.reduce_dim
        self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
        if self.params.origindataset=='cub':
            self.dcov1 = BDC(is_vec=True, input_dim=self.feature.feat_dim[2], dimension_reduction=reduce_dim)
            self.dcov2 = BDC(is_vec=True, input_dim=self.feature.feat_dim[1], dimension_reduction=reduce_dim)
            self.dcov3 = BDC(is_vec=True, input_dim=self.feature.feat_dim[0], dimension_reduction=reduce_dim)
        else:
            self.dcov1 = BDC(is_vec=True, input_dim=self.feature.feat_dim[0], dimension_reduction=reduce_dim)
            self.dcov2 = BDC(is_vec=True, input_dim=self.feature.feat_dim[1], dimension_reduction=reduce_dim)
            self.dcov3 = BDC(is_vec=True, input_dim=self.feature.feat_dim[2], dimension_reduction=reduce_dim)
    def feature_forward(self, x1,x2,x3):
        out1 = self.dcov1(x1)# bdc_module.forward  56
        out2 = self.dcov2(x2)# bdc_module.forward  56
        out3 = self.dcov3(x3)# bdc_module.forward  56
        return out1,out2,out3

    def set_forward(self, x, is_feature=False):
        z_support, z_query,y_support, y_query,x_support, x_query = self.parse_feature(x, is_feature)# ==》template.parse_ferature


        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)

        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        if self.n_support>1:
            scores1=self.setO(z_support,z_query)
            return scores1
        else:
            scores = self.metric(z_query, z_proto)
            return scores

    def get_feature(self, x, is_feature=False):
        x1,x2,x3=self.get_feature_X(x, is_feature)
        return x1,x2,x3

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
        loss1=self.loss_fn(scores, y_query)

        return float(top1_correct),float(top1_correct2),float(top1_correct3), len(y_label), loss1, scores


    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        if self.n_support > 1:

            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            score = (x * y).sum(2)

        return score

    def setO(self,supports,querys):
        T1 = supports.contiguous().unsqueeze(0).expand(self.n_way*self.n_query, -1, -1,-1)
        T2 = querys.contiguous().unsqueeze(1).expand(-1, self.n_support, -1)
        T2=T2.contiguous().unsqueeze(1).expand(-1, self.n_support,-1,-1)
        query=querys.contiguous().unsqueeze(1).expand(-1, 5, -1)
        dist = torch.pow(T1 - T2, 2).sum(3)
        score = -dist
        temperature = self.params.tem
        softmax_output2=F.softmax(score*temperature,dim=2)
        new_scores2=torch.pow(torch.sum(T1*softmax_output2.unsqueeze(3),dim=2)-query,2).sum(2)
        return  -new_scores2
