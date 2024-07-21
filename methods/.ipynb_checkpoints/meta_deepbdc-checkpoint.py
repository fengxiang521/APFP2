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

    #这段代码是一个名为set_forward的函数，它接受一个输入x和一个布尔值is_feature，并返回一个分数张量。
    # 在函数中，它首先调用parse_feature函数来获取支持集和查询集的特征表示。
    # 然后，它将支持集的特征表示重塑为一个三维张量，其中第一维表示类别，第二维表示支持集中每个类别的样本数，第三维表示特征维度。
    # 接下来，它计算每个类别的原型向量，即支持集中每个类别的特征表示的平均值。
    # 然后，它将查询集的特征表示重塑为一个二维张量，其中第一维表示查询样本，第二维表示特征维度。
    # 最后，它调用metric函数来计算查询样本和原型向量之间的相似度得分，并将它们作为分数张量返回。
    def set_forward(self, x, is_feature=False):
        z_support, z_query,y_support, y_query,x_support, x_query = self.parse_feature(x, is_feature)# ==》template.parse_ferature


        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        #y_proto = y_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        #x_proto = x_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)

        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        #y_query = y_query.contiguous().view(self.n_way * self.n_query, -1)
        #x_query = x_query.contiguous().view(self.n_way * self.n_query, -1)

        #scores = self.metric(z_query, z_proto)
        if self.n_support>1:
            scores1,new_scores1,new_scores2,new_scores3,new_scores4,PC_scores1,PC_scores2=self.setO(z_support,z_query)
            return new_scores2, new_scores1, scores1 
        else:
            scores = self.metric(z_query, z_proto)
            return scores, scores, scores

    def get_feature(self, x, is_feature=False):
        x1,x2,x3=self.get_feature_X(x, is_feature)
        return x1,x2,x3

    #这段代码是一个名为set_forward_loss的函数，它接受一个输入x，并返回四个值。
    # 首先，它创建一个y_query张量，其中包含重复的range(self.n_way)，重复self.n_query次。
    # 然后，它创建一个y_label数组，其中包含重复的range(self.n_way)，重复self.n_query次。
    # 接下来，它调用set_forward函数来计算分数。
    # 然后，它使用topk函数找到每个查询样本的最高分数和相应的标签。
    # 最后，它计算正确的预测数，总样本数，交叉熵损失和分数，并将它们作为元组返回。

    #换句话说，这个函数计算了一个元组，其中包含了模型在给定输入x的情况下的性能指标。
    # 其中包括正确预测的数量，总样本数，交叉熵损失和分数。
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
        #print(top1_correct)
        #print(top1_correct2)
        #print(top1_correct3)
        loss1=self.loss_fn(scores, y_query)

        return float(top1_correct),float(top1_correct2),float(top1_correct3), len(y_label), loss1, scores

    #这段代码定义了一个名为metric的函数，它计算两个张量x和y之间的相似度得分。
    # x和y都是二维张量，其中第一维表示样本，第二维表示特征维度。
    # 该函数首先获取x和y的大小，然后将x和y扩展为三维张量，其中第一维表示x中的样本，第二维表示y中的样本，第三维表示特征维度。
    # 然后，如果支持集中的样本数大于1，则计算欧几里得距离的平方，并将其取负作为相似度得分。
    # 否则，它计算x和y之间的点积，并将其作为相似度得分返回。
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
        z_proto = supports.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_proto=z_proto.contiguous().unsqueeze(0).expand(self.n_way*self.n_query, -1, -1)
        Pscores=-(torch.pow(query-z_proto,2).sum(2))
        dist = torch.pow(T1 - T2, 2).sum(3)
        score = -dist
        softmax_output=F.softmax(score,dim=2)
        temperature = 0.007
        softmax_output1=F.softmax(score*temperature,dim=2)
        temperature = 0.008
        softmax_output2=F.softmax(score*temperature,dim=2)
        temperature = 0.01
        softmax_output3=F.softmax(score*temperature,dim=2)
        temperature = 0.005
        softmax_output4=F.softmax(score*temperature,dim=2)
        proto2=torch.sum(T1*softmax_output1.unsqueeze(3),dim=2)
        new_scores1=torch.pow(proto2-query,2).sum(2)

        #new_scores2=torch.mul(softmax_output2,score).sum(2)
        new_scores2=torch.pow(torch.sum(T1*softmax_output2.unsqueeze(3),dim=2)-query,2).sum(2)
        #new_scores3=torch.mul(softmax_output3,score).sum(2)
        new_scores3=torch.pow(torch.sum(T1*softmax_output3.unsqueeze(3),dim=2)-query,2).sum(2)

        #new_scores4=torch.mul(softmax_output4,score).sum(2)
        new_scores4=torch.pow(torch.sum(T1*softmax_output4.unsqueeze(3),dim=2)-query,2).sum(2)

        PCproto1,PCproto2,weight2=self.setProto(supports)
        PCprotox1=PCproto1.unsqueeze(0).expand(self.n_way*self.n_query, -1, -1)
        PCprotox2=PCproto2.unsqueeze(0).expand(self.n_way*self.n_query, -1, -1)
        PCscores1=-(torch.pow(PCprotox1-query,2).sum(2))
        PCscores2=-(torch.pow(PCprotox2-query,2).sum(2))
        #计算各支持集样本与加权原型样本距离
        # scores_X=-torch.pow(PCproto2.unsqueeze(1).expand(-1, 5, -1)-supports,2).sum(2)
        # scores_X=scores_X.unsqueeze(0).expand(75,-1,-1)
        # score=score*weight2
        # temperature = 0.005
        # softmax_output4 = F.softmax(score * temperature, dim=2)
        # softmax_output4=(softmax_output4+weight2)/2
        # new_scores1 = torch.pow(torch.sum(T1 * softmax_output4.unsqueeze(3), dim=2) - query, 2).sum(2)
        sum_dim_3 = torch.abs(dist-torch.mean(dist, dim=2,keepdim=True))
        percentage_tensor = sum_dim_3 / torch.sum(sum_dim_3,dim=2,keepdim=True)
        temperature=0.072
        weight = F.softmax(-percentage_tensor * temperature, dim=2)
        new_scores1 = torch.pow(torch.sum(T1 * ((softmax_output2+weight)/2).unsqueeze(3), dim=2) - query, 2).sum(2)

        return  Pscores,-new_scores1,-new_scores2,-new_scores3,-new_scores4,PCscores1,PCscores2

    def XXX(self,scores,proto):
        T1=proto.contiguous().unsqueeze(0).expand(5, -1, -1)
        T2=proto.contiguous().unsqueeze(1).expand(-1, 5, -1)
        support=-1*(torch.pow(T1 - T2, 2).sum(2))
        rescores = torch.zeros(75, 5).cuda()
        for i in range(75):
            for j in range(5):
                rescores[i][j]=scores[i][j]
        return scores


    def setProto(self,support):
        # 获取张量的维度
        #batch_size, num_samples, feature_dim = support.shape

        # 扩展张量，使其在第二个维度上与其他样本相减
        expanded_tensor1 = support.unsqueeze(1).expand(-1, 5, -1, -1)
        expanded_tensor2 = support.unsqueeze(2).expand(-1, -1, 5, -1)


        # 计算差值的绝对值
        abs_diff = torch.abs(expanded_tensor1 - expanded_tensor2).sum(3).sum(2)/4

        # 沿特征维度求和
        dist=torch.pow(expanded_tensor1-expanded_tensor2,2).sum(3).sum(2)/4
        temp = 0.004
        weight2=F.softmax(-dist*temp)
        proto1=torch.sum(support*weight2.unsqueeze(2),dim=1)
        temp=0.005
        weight2 = F.softmax(-dist * temp)
        proto2 = torch.sum(support * weight2.unsqueeze(2), dim=1)


        return proto1,proto2,weight2





    def kendalltau_distence(self,queries,supports):
        p_tensor = queries.unsqueeze(1).expand(-1, 5, -1)
        q_tensor = supports.unsqueeze(0).expand(p_tensor.size(0), -1, -1)

        mask_matrix = ~torch.eye(5, dtype=torch.bool).unsqueeze(0).expand(p_tensor.size(0), -1, -1)
        p_tensor = p_tensor[mask_matrix].view(p_tensor.size(0), 5, 5 - 1)

        q_tensor = q_tensor[mask_matrix].view(p_tensor.size(0), 5, 5 - 1)

        p_tensor *= (torch.sum(q_tensor, dim=2, keepdim=True) / torch.sum(p_tensor, dim=2, keepdim=True))
        p_tensor = (p_tensor + q_tensor) / 2
        # p_tensor = self.calculate_probabilities(p_tensor)
        # q_tensor = self.calculate_probabilities(q_tensor)
        js_distances = torch.zeros(75, 5).cuda()
        for i in range(75):
            for j in range(5):
                p = p_tensor[i][j]
                q = q_tensor[i][j]
                js_distance = self.jensen_shannon_divergence(p, q)
                js_distances[i, j] = js_distance
        return js_distances

    def jensen_shannon_divergence(self,p, q):
        m = 0.5 * (p + q)
        kl_p = torch.sum(p * torch.log(p / m))
        kl_q = torch.sum(q * torch.log(q / m))
        return 0.5 * (kl_p + kl_q)

    #Hellinger 距离
    def calculate_probabilities(self,values):
        values_exp = torch.sum(values,dim=2,keepdim=True)
        probabilities = values/values_exp
        return probabilities

    def hellinger_distance(self,p_tensor, q_tensor):
        p_probs = self.calculate_probabilities(p_tensor)
        q_probs = self.calculate_probabilities(q_tensor)
        js_distances = torch.zeros(75, 5).cuda()
        HL_distances = torch.zeros(75, 5).cuda()
        CS_distances = torch.zeros(75, 5).cuda()
        EN_distances = torch.zeros(75, 5).cuda()
        TV_distances = torch.zeros(75, 5).cuda()
        KL_distances = torch.zeros(75, 5).cuda()
        for i in range(75):
            for j in range(5):
                p = p_probs[i]
                q = q_probs[j]
                h_distance = torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2))
                js_distances[i, j] = h_distance
            mean = torch.mean(js_distances[i])
            squared_diff = (js_distances[i] - mean) ** 2
            variance = torch.mean(squared_diff)
            for k in range(5):
                max_value, max_index = torch.max(q_probs[k], dim=0)
                q1 = torch.cat((q_probs[k][:max_index], q_probs[k][max_index + 1:]))
                p1 = torch.cat((p_probs[i][:max_index], p_probs[i][max_index + 1:]))
                hl = torch.sqrt(0.5 * torch.sum((torch.sqrt(p1) - torch.sqrt(q1)) ** 2))
                cs= torch.sum(((p1 - q1) ** 2) / p1)
                en= torch.sum((torch.sqrt(p1) - torch.sqrt(q1)) ** 2) / 2.0
                tv=0.5 * torch.sum(torch.abs(p1 - q1))
                kl=self.jensen_shannon_divergence(p1, q1)
                HL_distances[i, k] = hl
                CS_distances[i, k] = cs
                EN_distances[i, k] = en
                TV_distances[i, k] = tv
                KL_distances[i, k] = kl

        return js_distances,HL_distances,CS_distances,EN_distances,TV_distances,KL_distances
    #Total Variation 散度
    def total_variation_divergence(self,p_tensor, q_tensor):
        p_probs = self.calculate_probabilities(p_tensor)
        q_probs = self.calculate_probabilities(q_tensor)
        js_distances = torch.zeros(75, 5).cuda()
        for i in range(75):
            for j in range(5):
                p = p_probs[i]
                q = q_probs[j]
                tv_divergence = 0.5 * torch.sum(torch.abs(p - q))
                js_distances[i, j] = tv_divergence
        return js_distances
    #能量散度
    def energy_divergence(self,p_tensor, q_tensor):
        p_probs = self.calculate_probabilities(p_tensor)
        q_probs = self.calculate_probabilities(q_tensor)
        js_distances = torch.zeros(75, 5).cuda()
        for i in range(75):
            for j in range(5):
                p = p_probs[i]
                q = q_probs[j]
                energy_divergence = torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2) / 2.0
                js_distances[i, j] = energy_divergence
        return js_distances

    #卡方距离
    def chi_squared_distance(self,p_tensor, q_tensor):
        p_probs = self.calculate_probabilities(p_tensor)
        q_probs = self.calculate_probabilities(q_tensor)
        js_distances = torch.zeros(75, 5).cuda()
        for i in range(75):
            for j in range(5):
                p = p_probs[i]
                q = q_probs[j]
                chi_squared_divergence = torch.sum(((p - q) ** 2) / p)
                js_distances[i, j] = chi_squared_divergence
        return js_distances

    #欧式距离
    def oushi(self,p_tensor,q_tensor):
        n=p_tensor.size(0)
        m=q_tensor.size(0)
        p_tensor = self.calculate_probabilities(p_tensor)
        q_tensor = self.calculate_probabilities(q_tensor)
        js_distances = torch.zeros(n, m).cuda()
        HL_distances = torch.zeros(n, m).cuda()
        CS_distances = torch.zeros(n, m).cuda()
        EN_distances = torch.zeros(n, m).cuda()
        TV_distances = torch.zeros(n, m).cuda()
        KL_distances = torch.zeros(n, m).cuda()
        PC_distances = torch.zeros(n, m).cuda()
        WD_distances = torch.zeros(n, m).cuda()
        CD_distances = torch.zeros(n, m).cuda()
        EMD_distances = torch.zeros(n, m).cuda()
        BD_distances = torch.zeros(n, m).cuda()
        MMD_distances = torch.zeros(n, m).cuda()
        SF_distances = torch.zeros(n, m).cuda()
        GS_distances = torch.zeros(n, m).cuda()
        for i in range(n):
            for k in range(m):
                max_value, max_index = torch.min(q_tensor[k], dim=0)
                q1 = torch.cat((q_tensor[k][:max_index], q_tensor[k][max_index + 1:])).cpu()
                p1 = torch.cat((p_tensor[i][:max_index], p_tensor[i][max_index + 1:])).cpu()
                hl = torch.sqrt(0.5 * torch.sum((torch.sqrt(p1) - torch.sqrt(q1)) ** 2))
                cs = torch.sum(((p1 - q1) ** 2) / p1)
                en = torch.sum((torch.sqrt(p1) - torch.sqrt(q1)) ** 2) / 2.0
                tv = 0.5 * torch.sum(torch.abs(p1 - q1))
                kl = self.jensen_shannon_divergence(p1, q1)
                ####Pearson相关系数####
                p_mean = torch.mean(p1)
                q_mean = torch.mean(q1)
                p_std = torch.std(p1)
                q_std = torch.std(q1)
                covariance = torch.mean((p1 - p_mean) * (q1 - q_mean))
                pc = covariance / (p_std * q_std)
                #####Wasserstein距离####
                p_array = p1.detach().numpy()
                q_array = q1.detach().numpy()
                p_cov = np.cov(p_array)
                q_cov = np.cov(q_array)
                wd = wasserstein_distance(p_array, q_array)
                #计算Cramer距离
                #cs = energy_distance(p_array, q_array)
                cd = energy_distance(p_array, q_array)
                #emd = wasserstein_distance(p1, q1)
                emd = 0
                bd=0
                # bd = 0.125 * (q_mean - p_mean) ** 2 + 0.125 * np.trace(p_cov + q_cov - 2 * np.sqrt(
                #     np.matmul(np.matmul(np.linalg.inv(p_cov), np.matmul(q_cov, p_cov)), np.linalg.inv(q_cov))))
                #mmd = np.abs(p_mean.detach.numpy() - q_mean.detach.numpy())
                mmd = 0
                transformed_p = p_array * 2
                transformed_q = q_array * 3
                sf = np.abs(transformed_p - transformed_q).mean()
                ## 使用高斯核函数计算相似性，示例中的gamma为核函数参数
                gamma = 0.1
                similarity_matrix = rbf_kernel(p_array.reshape(-1, 1), q_array.reshape(-1, 1), gamma=gamma)
                gs = np.mean(similarity_matrix)

                HL_distances[i, k] = hl
                CS_distances[i, k] = cs
                EN_distances[i, k] = en
                TV_distances[i, k] = tv
                KL_distances[i, k] = kl
                PC_distances[i, k] = pc
                WD_distances[i, k] = wd
                CD_distances[i, k] = cd
                EMD_distances[i, k] = emd
                BD_distances[i, k] = bd
                MMD_distances[i, k] = mmd
                SF_distances[i, k] = torch.tensor(sf)
                GS_distances[i, k] = torch.tensor(gs)
        return js_distances, HL_distances, CS_distances, EN_distances, TV_distances, KL_distances,PC_distances,WD_distances,CD_distances,EMD_distances,BD_distances,MMD_distances,SF_distances,GS_distances

    def oushi1(self, p_tensor, q_tensor):


        p_tensor=p_tensor.unsqueeze(1).expand(-1, 5, -1)
        q_tensor = q_tensor.unsqueeze(0).expand(p_tensor.size(0), -1, -1)



        mask_matrix = ~torch.eye(5, dtype=torch.bool).unsqueeze(0).expand(p_tensor.size(0), -1, -1)
        p_tensor = p_tensor[mask_matrix].view(p_tensor.size(0), 5, 5 - 1)


        q_tensor = q_tensor[mask_matrix].view(p_tensor.size(0), 5, 5 - 1)

        p_tensor*=(torch.sum(q_tensor,dim=2,keepdim=True)/torch.sum(p_tensor,dim=2,keepdim=True))
        p_tensor=(p_tensor+q_tensor)/2
        p_tensor = self.calculate_probabilities(p_tensor)
        q_tensor = self.calculate_probabilities(q_tensor)



        hl = torch.sqrt(0.5 * torch.sum((torch.sqrt(p_tensor) - torch.sqrt(q_tensor)) ** 2, dim=2))
        cs = torch.sum(((p_tensor - q_tensor) ** 2) / p_tensor, dim=2)
        en = torch.sum((torch.sqrt(p_tensor) - torch.sqrt(q_tensor)) ** 2, dim=2) / 2.0
        tv = 0.5 * torch.sum(torch.abs(p_tensor - q_tensor), dim=2)
        kl = self.js_divergence(p_tensor, q_tensor)
        #pc = torch.sum((p_tensor - p_tensor.mean(dim=1, keepdim=True)) * (q_tensor - q_tensor.mean(dim=1, keepdim=True)), dim=2) / (
        #             p_tensor.std(dim=1, unbiased=False) * q_tensor.std(dim=1, unbiased=False))
        #
        # p_array = p1.detach().cpu().numpy()
        # q_array = q1.detach().cpu().numpy()
        # wd = np.array([wasserstein_distance(p, q) for p, q in zip(p_array, q_array)])
        # cd = np.array([energy_distance(p, q) for p, q in zip(p_array, q_array)])

        # Initialize other distance tensors similarly
        ou=torch.pow(p_tensor - q_tensor, 2).sum(2)
        return  hl, cs, en, tv, kl,ou#, pc, wd, cd

    def kl_divergence(self,p, q):
        return (p * torch.log(p / q)).sum(dim=-1)

    def js_divergence(self,tensor1, tensor2):
        mid_tensor = 0.5 * (tensor1 + tensor2)
        kl_div1 = self.kl_divergence(tensor1, mid_tensor)
        kl_div2 = self.kl_divergence(tensor2, mid_tensor)
        js_div = 0.5 * (kl_div1 + kl_div2)
        return js_div