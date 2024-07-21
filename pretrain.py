import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os
import argparse

from data.datamgr import SimpleDataManager, SetDataManager
from methods.template import BaselineTrain
from utils import *


def train(params, base_loader, val_loader, model, stop_epoch):
    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0
    if params.method in ['stl_deepbdc', 'meta_deepbdc']:
        bas_params1 = filter(lambda p: id(p) not in [id(model.dcov1.temperature), id(model.dcov2.temperature),id(model.dcov3.temperature)], model.parameters())
        optimizer = torch.optim.SGD([
            {'params': bas_params1},
            {'params': model.dcov1.temperature, 'lr': params.t_lr},
            {'params': model.dcov2.temperature, 'lr': params.t_lr},
            {'params': model.dcov3.temperature, 'lr': params.t_lr}], lr=params.lr, weight_decay=params.wd, nesterov=True, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, weight_decay=params.wd, nesterov=True, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=params.milestones, gamma=params.gamma)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    for epoch in range(0, stop_epoch):
        start = time.time()
        model.train()
        trainObj1, top1,trainObj2, top2,trainObj3, top3 = model.train_loop(epoch, base_loader, optimizer)

        if params.dataset == 'tiered_imagenet':
            if epoch >= params.milestones[1]:
                model.eval()
                if params.val in ['meta']:
                    if params.val == 'meta':
                        valObj, acc = model.meta_test_loop(val_loader)
                    trlog['val_loss'].append(valObj)
                    trlog['val_acc'].append(acc)
                    if acc > trlog['max_acc']:
                        print("best model! save...")
                        trlog['max_acc'] = acc
                        trlog['max_acc_epoch'] = epoch
                        outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                        torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
                    print("val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
                    print("model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))
        else:
            model.eval()
            if params.val in ['meta']:
                if params.val == 'meta':
                    valObj, acc = model.meta_test_loop(val_loader)
                trlog['val_loss'].append(valObj)
                trlog['val_acc'].append(acc)
                if acc > trlog['max_acc']:
                    print("best model! save...")
                    trlog['max_acc'] = acc
                    trlog['max_acc_epoch'] = epoch
                    outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                    torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
                print("val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
                print("model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))
        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        trlog['train_loss'].append(trainObj1)
        trlog['train_acc'].append(top1)

        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))
        lr_scheduler.step()

        print("This epoch use %.2f minutes" % ((time.time() - start) / 60))
        print("train loss is {:.2f}, train acc is {:.2f}".format(trainObj1, top1))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', default=224, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
    parser.add_argument('--batch_size', default=64, type=int, help='pre-training batch size')
    parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate of the backbone')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--t_lr', type=float, default=0.05, help='initial learning rate uesd for the temperature of bdc module')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--milestones', nargs='+', type=int, default=[100, 150], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=170, type=int, help='stopping epoch')
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--dataset', default='cub', choices=['mini_imagenet','tiered_imagenet','cub'])
    parser.add_argument('--data_path', type=str, help='dataset path',default='../SetBDC/filelist/cub/')

    parser.add_argument('--model', default='ResNet18', choices=['ResNet12', 'ResNet18'])
    parser.add_argument('--method', default='protonet', choices=['meta_deepbdc', 'stl_deepbdc', 'protonet', 'good_embed'])
    
    parser.add_argument('--val', default='meta', choices=['meta', 'last'], help='validation method')
    parser.add_argument('--val_n_episode', default=600, type=int, help='number of episodes in meta validation')
    parser.add_argument('--val_n_way', default=5, type=int, help='number of  classes to classify in meta validation')
    parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class, same as n_support during meta validation')
    parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

    #parser.add_argument('--extra_dir', default='CAO', help='recording additional information')
    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in training')
    parser.add_argument('--save_freq', default=50, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimensions of BDC dimensionality reduction layer')
    parser.add_argument('--dropout_rate', default=0.6, type=float, help='dropout rate for pretrain and distillation')
    parser.add_argument('--tem', default=0.08, type=float)

    params = parser.parse_args()

    num_gpu = set_gpu(params)
    set_seed(params.seed)


    if params.val == 'last':
        val_file = None
    elif params.val == 'meta':
        val_file = 'val'

    json_file_read = False
    if params.dataset == 'mini_imagenet':
        base_file = 'train'
        params.num_classes = 64
    elif params.dataset == 'cub':
        base_file = 'base.json'
        val_file =  'val.json'
        json_file_read = True
        params.num_classes = 200
    elif params.dataset == 'tiered_imagenet':
        base_file = 'train'
        params.num_classes = 351
    else:
        ValueError('dataset error')

    base_datamgr = SimpleDataManager(params.data_path, params.image_size, batch_size=params.batch_size, json_read=json_file_read)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    if params.val == 'meta':
        test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    else:
        val_loader = None

    model = BaselineTrain(params, model_dict[params.model], params.num_classes)

    model = model.cuda()

    params.checkpoint_dir = './checkpoints/%s/%s_%s' % (params.dataset, params.model, params.method)
    params.checkpoint_dir += '_pretrain'

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(params)
    model = train(params, base_loader, val_loader, model, params.epoch)
