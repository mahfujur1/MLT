from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import init

from mlt.utils.rerank import compute_jaccard_dist

from mlt import datasets
from mlt import models
from mlt.trainers import DbscanBaseTrainer
from mlt.evaluators import Evaluator, extract_features
from mlt.utils.data import IterLoader
from mlt.utils.data import transforms as T
from mlt.utils.data.sampler import RandomMultipleGallerySampler
from mlt.utils.data.preprocessor import Preprocessor
from mlt.utils.logging import Logger
from mlt.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from mlt.NCE.NCEAverage import MemoryMoCo
from mlt.NCE.NCEAverage import MemoryMoCo_id
from mlt.utils.faiss_rerank import compute_jaccard_distance
# import ipdb


start_epoch = best_mAP = 0

def get_data(name, data_dir, l=1):
    root = osp.join(data_dir)

    dataset = datasets.create(name, root, l)

    label_dict = {}
    for i, item_l in enumerate(dataset.train):
        if item_l[1] in label_dict:
            label_dict[item_l[1]].append(i)
        else:
            label_dict[item_l[1]] = [i]
    return dataset, label_dict


def get_train_loader(dataset, height, width, choice_c, batch_size, workers,
                     num_instances, iters, trainset=None):
    mean_im=[0.485, 0.456, 0.406]
    normalizer = T.Normalize(mean=mean_im,
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.596, 0.558, 0.497])
    ])

    train_set = trainset #dataset.train if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances, choice_c)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                transform=train_transformer, mutual=True),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    # train_loader = IterLoader(
    #     DataLoader(UnsupervisedCamStylePreprocessor(train_set, root=dataset.images_dir, transform=train_transformer,
    #                                                 num_cam=dataset.num_cam,camstyle_dir=dataset.camstyle_dir, mutual=True),
    #                batch_size=batch_size, num_workers=0, sampler=sampler,#workers
    #                shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model_dbscan(args,ncs):
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout,
                            num_classes=ncs)
    model_1.cuda()
    model_1 = nn.DataParallel(model_1)
    # initial_weights = load_checkpoint(args.init_1)
    # copy_state_dict(initial_weights['state_dict'], model_1)
    return model_1


def create_model(args, ncs):

    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout,
                            num_classes=ncs)
    #model_2 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)

    model_1_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout,
                               num_classes=ncs)
    #model_2_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout,
    #                            num_classes=args.num_clusters)

    model_1.cuda()
    #model_2.cuda()
    model_1_ema.cuda()
    #model_2_ema.cuda()
    model_1 = nn.DataParallel(model_1)
    #model_2 = nn.DataParallel(model_2)
    model_1_ema = nn.DataParallel(model_1_ema)
    #model_2_ema = nn.DataParallel(model_2_ema)

    # initial_weights = load_checkpoint(args.init_1)
    # copy_state_dict(initial_weights['state_dict'], model_1)
    # copy_state_dict(initial_weights['state_dict'], model_1_ema)
    # for i,cl in enumerate(ncs):
    #     exec('model_1_ema.module.classifier{}_{}.weight.data.copy_(model_1.module.classifier{}_{}.weight.data)'.format(i,cl,i,cl))

    # initial_weights = load_checkpoint(args.init_2)
    # copy_state_dict(initial_weights['state_dict'], model_2)
    # copy_state_dict(initial_weights['state_dict'], model_2_ema)
    # model_2_ema.module.classifier.weight.data.copy_(model_2.module.classifier.weight.data)

    # for param in model_1_ema.parameters():
    #     param.detach_()
    # for param in model_2_ema.parameters():
    #     param.detach_()

    return model_1, None, model_1_ema, None#model_1, model_2, model_1_ema, model_2_ema


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

import sinkhornknopp as sk
class Optimizer:
    def __init__(self, target_label, m, dis_gt, t_loader,N, hc=3, ncl=None,  n_epochs=200,
                 weight_decay=1e-5, ckpt_dir='/',fc_len=3500):
        self.num_epochs = n_epochs
        self.momentum = 0.9
        self.weight_decay = weight_decay
        self.checkpoint_dir = ckpt_dir
        self.N=N
        self.resume = True
        self.checkpoint_dir = None
        self.writer = None
        # model stuff
        self.hc = len(ncl)#10
        self.K = ncl#3000
        self.K_c =[fc_len for _ in range(len(ncl))]
        self.model = m
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L = [torch.LongTensor(target_label[i]).to(self.dev) for i in range(len(self.K))]
        self.nmodel_gpus = 4#len()
        self.pseudo_loader = t_loader#torch.utils.data.DataLoader(t_loader,batch_size=256)
        # can also be DataLoader with less aug.
        self.train_loader = t_loader
        self.lamb = 25#args.lamb # the parameter lambda in the SK algorithm
        self.cpu=True
        self.dis_gt=dis_gt
        dtype_='f64'
        if dtype_ == 'f32':
            self.dtype = torch.float32 if not self.cpu else np.float32
        else:
            self.dtype = torch.float64 if not self.cpu else np.float64

        self.outs = self.K
        # activations of previous to last layer to be saved if using multiple heads.
        self.presize =  2048#4096 #

    def optimize_labels(self):
        if self.cpu:
            sk.cpu_sk(self)
        else:
            sk.gpu_sk(self)

        # save Label-assignments: optional
        # torch.save(self.L, os.path.join(self.checkpoint_dir, 'L', str(niter) + '_L.gz'))

        # free memory
        data = 0
        self.PS = 0

        return self.L

import collections
import os

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def write_sta_im(train_loader):
    label2num=collections.defaultdict(int)
    save_label=[]
    for x in train_loader:
        label2num[x[1]]+=1
        save_label.append(x[1])
    labels=sorted(label2num.items(),key=lambda item:item[1])[::-1]
    num = [j for i, j in labels]
    distribution = np.array(num)/len(train_loader)

    return num,save_label
def print_cluster_acc(label_dict,target_label_tmp):
    num_correct = 0
    for pid in label_dict:
        pid_index = np.asarray(label_dict[pid])
        pred_label = np.argmax(np.bincount(target_label_tmp[pid_index]))
        num_correct += (target_label_tmp[pid_index] == pred_label).astype(np.float32).sum()
    cluster_accuracy = num_correct / len(target_label_tmp)
    print(f'cluster accucary: {cluster_accuracy:.3f}')

def kmeans_cluster(f):


    pass

def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters > 0) else None
    ncs = [int(x) for x in args.ncs.split(',')]
    ncs_dbscan=ncs.copy()
    dataset_target, label_dict = get_data(args.dataset_target, args.data_dir, len(ncs))
    dataset_source, _ = get_data(args.dataset_source, args.data_dir, len(ncs))

    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)

    # cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,
    #                                  testset=dataset_target.train)
    tar_cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,
                                         testset=dataset_target.train)
    sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width, args.batch_size, args.workers,
                                          testset=dataset_source.train)
    distribution,save_label = write_sta_im(dataset_source.train)

    distribution_t,save_label_t = write_sta_im(dataset_target.train)

    np.save("duke_dis.npy", distribution)
    np.save("market_dis.npy", distribution_t)
    np.save("duke_label.npy", save_label)
    np.save("market_label.npy", save_label_t)
    print("non")
    # Create model
    model = create_model_dbscan(args, ncs)

    clsuter_style = 'dbscan'
    #target_label = np.load("target_label.npy")
    epoch = 0
    target_features, _ = extract_features(model, tar_cluster_loader, print_freq=100)

    target_features =  torch.stack(list(target_features.values()))#torch.cat([target_features[f[0]].unsqueeze(0) for f in dataset_target.train], 0)
    target_features = F.normalize(target_features, dim=1)
    # Calculate distance
    print('==> Create pseudo labels for unlabeled target domain')

    rerank_dist = compute_jaccard_distance(target_features, k1=args.k1, k2=args.k2)
    del target_features

    if (epoch == 0):
        # DBSCAN cluster
        eps = 0.6  # 0.6
        delta=0.02

        eps_l=0.6-delta
        eps_r=0.6+delta
        print('Clustering criterion: eps: {:.3f}, eps_l: {:.3f}, eps_r: {:.3f}'.format(eps,eps_l,eps_r))
        cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
        cluster_l = DBSCAN(eps=eps_l, min_samples=4, metric='precomputed', n_jobs=-1)
        cluster_r = DBSCAN(eps=eps_r, min_samples=4, metric='precomputed', n_jobs=-1)

    # select & cluster images as training set of this epochs
    pseudo_labels = cluster.fit_predict(rerank_dist)
    pseudo_labels_l = cluster_l.fit_predict(rerank_dist)
    pseudo_labels_r = cluster_r.fit_predict(rerank_dist)

    num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    num_ids_l = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    num_ids_r = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

    p1=[]
    p2=[]
    p3=[]
    new_dataset=[]
    for i, (item, label,label1,label2) in enumerate(zip(dataset_target.train, pseudo_labels,pseudo_labels_l,pseudo_labels_r)):
        if label == -1 and label1==-1 and label2==-1:
            continue
        if label == -1:
            label=num_ids
            num_ids+=1
        if label1 == -1:
            label1 = num_ids_l
            num_ids_l += 1
        if label2 == -1:
            label2 = num_ids_r
            num_ids_r += 1
        p1.append(label)
        p2.append(label1)
        p3.append(label2)
        new_dataset.append((item[0], label,label1,label2, item[-1]))
    target_label = [p1,p2,p3]
    p11 = len(set(p1)) + 1
    p22 = len(set(p2)) + 1
    p33 = len(set(p3)) + 1
    ncs = [p11,p22,p33]
    print('new class are {}, length of new dataset is {}'.format(ncs,len(new_dataset)))
    # for i, num_cluster in enumerate(ncs):
    #     exec("model_1.classifier{}_{} = nn.Linear(model_1.num_features, {}, bias=False)".format(i, num_cluster,
    #                                                                                          num_cluster))
    #     exec("model_1.init.normal_(model_1.classifier{}_{}.weight, std=0.001)".format(i, num_cluster))
    #     exec("model_1_ema.classifier{}_{} = nn.Linear(model_1_ema.num_features, {}, bias=False)".format(i, num_cluster,
    #                                                                                          num_cluster))
    #     exec("model_1_ema.init.normal_(model_1_ema.classifier{}_{}.weight, std=0.001)".format(i, num_cluster))
    # Evaluator
    fc_len=3500
    model_1, _, model_1_ema, _ = create_model(args, [fc_len for _ in range(len(ncs))])
    evaluator_1 = Evaluator(model_1)
    evaluator_1_ema = Evaluator(model_1_ema)

    # evaluator_1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
    #                      cmc_flag=True)
    clusters = [args.num_clusters] * args.epochs# TODO: dropout clusters



    contrast = MemoryMoCo_id(2048, len(new_dataset), K=8192,
                             index2label=target_label, choice_c=args.choice_c, T=0.07,
                             use_softmax=True).cuda()

    tar_selflabel_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,
                                         testset=new_dataset)

    o = Optimizer(target_label, dis_gt=distribution, m=model_1, ncl=ncs,
                  t_loader=tar_selflabel_loader, N=len(new_dataset),fc_len=fc_len)

    for epoch in range(len(clusters)):
        iters_ = 300 if epoch  % 1== 0 else iters
        if epoch % 1 == 0 and epoch != 0:
            target_features, _ = extract_features(model_1_ema, tar_cluster_loader, print_freq=50)
            target_features = torch.stack(list(
                target_features.values()))  # torch.cat([target_features[f[0]].unsqueeze(0) for f in dataset_target.train], 0)
            target_features = F.normalize(target_features, dim=1)
            # Calculate distance
            print('==> Create pseudo labels for unlabeled target domain with')
            rerank_dist = compute_jaccard_distance(target_features, k1=args.k1, k2=args.k2)
            del target_features

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            pseudo_labels_l = cluster_l.fit_predict(rerank_dist)
            pseudo_labels_r = cluster_r.fit_predict(rerank_dist)

            num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            num_ids_l = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            num_ids_r = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

            p1 = []
            p2 = []
            p3 = []
            new_dataset = []

            for i, (item, label, label1, label2) in enumerate(
                    zip(dataset_target.train, pseudo_labels, pseudo_labels_l, pseudo_labels_r)):
                if label == -1 and label1 == -1 and label2 == -1:
                    continue
                if label == -1:
                    label = num_ids
                    num_ids += 1
                if label1 == -1:
                    label1 = num_ids_l
                    num_ids_l += 1
                if label2 == -1:
                    label2 = num_ids_r
                    num_ids_r += 1
                p1.append(label)
                p2.append(label1)
                p3.append(label2)
                new_dataset.append((item[0], label, label1, label2, item[-1]))
            target_label = [p1, p2, p3]
            p11 = len(set(p1)) + 1
            p22 = len(set(p2)) + 1
            p33 = len(set(p3)) + 1
            ncs = [p11, p22, p33]
            print('new class are {}, length of new dataset is {}'.format(ncs, len(new_dataset)))

            tar_selflabel_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,
                                                 testset=new_dataset)
            o = Optimizer(target_label, dis_gt=distribution, m=model_1, ncl=ncs,
                          t_loader=tar_selflabel_loader, N=len(new_dataset),fc_len=fc_len)


        target_label_o = o.L
        target_label = [np.asarray(target_label_o[i].data.cpu()) for i in range(len(ncs))]
        contrast.index2label = target_label
        '''
        #calculate cluster acc
        for i in range(3):
            for pid in label_dict:
                pid_index = np.asarray(label_dict[pid])
                pred_label = np.argmax(np.bincount(target_label700[pid_index]))
                num_correct += (target_label700[pid_index] == pred_label).astype(np.float32).sum()
            cluster_accuracy = num_correct / len(target_label700)
            print(f'cluster accucary: {cluster_accuracy:.3f}')
        '''

        # change pseudo labels
        for i in range(len(new_dataset)):
            new_dataset[i] = list(new_dataset[i])
            for j in range(len(ncs)):
                new_dataset[i][j+1] = int(target_label[j][i])
            new_dataset[i] = tuple(new_dataset[i])


        # print(nc,"============"+str(iters_))
        cc=args.choice_c#(args.choice_c+1)%len(ncs)
        train_loader_target = get_train_loader(dataset_target, args.height, args.width, cc,
                                               args.batch_size, args.workers, args.num_instances, iters_, new_dataset)

        # Optimizer
        params = []
        flag = 1.0
        # if 20<epoch<=40 or 60<epoch<=80 or 120<epoch:
        #     flag=0.1
        # else:
        #     flag=1.0

        for key, value in model_1.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr*flag, "weight_decay": args.weight_decay}]
        # for key, value in model_2.named_parameters():
        #     if not value.requires_grad:
        #         continue
        #     params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

        optimizer = torch.optim.Adam(params)

        # Trainer
        trainer = DbscanBaseTrainer(model_1, model_1_ema, contrast, None,
                             num_cluster=ncs, c_name=ncs,alpha=args.alpha, fc_len=fc_len)


        train_loader_target.new_epoch()

        # index2label = dict([(i, j) for i, j in enumerate(np.asarray(target_label[0]))])
        # index2label1= dict([(i, j) for i, j in enumerate(np.asarray(target_label[1]))])
        # index2label2 = dict([(i, j) for i, j in enumerate(np.asarray(target_label[2]))])


        trainer.train(epoch, train_loader_target, optimizer, args.choice_c,
                      ce_soft_weight=args.soft_ce_weight, tri_soft_weight=args.soft_tri_weight,
                      print_freq=args.print_freq, train_iters=iters_)

        if epoch>20:o.optimize_labels()

        # ecn.L = o.L

        # if nc ==yhua[-1]:
        #     while nc ==yhua[-1]:
        #         target_label_o = o.optimize_labels()
        #         yhua= yhua[:-1]

        def save_model(model_ema, is_best, best_mAP, mid):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model' + str(mid) + '_checkpoint.pth.tar'))
        if epoch==0:
            args.eval_step=2
        elif epoch==40:
            args.eval_step=1
        if ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            mAP_1 = 0#evaluator_1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                     #                        cmc_flag=False)

            mAP_2 = evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                                             cmc_flag=False)
            is_best = (mAP_1 > best_mAP) or (mAP_2 > best_mAP)
            best_mAP = max(mAP_1, mAP_2, best_mAP)
            save_model(model_1, (is_best), best_mAP, 1)
            save_model(model_1_ema, (is_best and (mAP_1 <= mAP_2)), best_mAP, 2)

            print('\n * Finished epoch {:3d}  model no.1 mAP: {:5.1%} model no.2 mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP_1, mAP_2, best_mAP, ' *' if is_best else ''))

    print('Test on the best model.')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model_1.load_state_dict(checkpoint['state_dict'])
    evaluator_1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mlt Training")
    # data
    parser.add_argument('-st', '--dataset-source', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-tt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--choice_c', type=int, default=1)
    parser.add_argument('--num-clusters', type=int, default=700)
    parser.add_argument('--ncs', type=str, default='58,60,62')

    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")

    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer

    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--iters', type=int, default=200)

    parser.add_argument('--lambda-value', type=float, default=0)
    # training configs

    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    parser.add_argument('--init-1', type=str, default='logs/dukemtmcTOmarket1501/resnet50-pretrain-1//model_best.pth.tar', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=2)
    parser.add_argument('--n-jobs', type=int, default=8)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/unsuper/'))
    print("======mlt_train_dbscan_self-labeling=======")


    main()
