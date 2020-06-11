from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

from sklearn.cluster import KMeans

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from mlt import datasets
from mlt import models
from mlt.trainers import mltTrainer_single
from mlt.evaluators import Evaluator, extract_features
from mlt.utils.data import IterLoader
from mlt.utils.data import transforms as T
from mlt.utils.data.sampler import RandomMultipleGallerySampler
from mlt.utils.data.preprocessor import Preprocessor
from mlt.utils.logging import Logger
from mlt.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from mlt.NCE.NCEAverage import MemoryMoCo_id

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
                     num_instances, iters):
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

    train_set = dataset.train
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


def create_model(args, ncs,mb_h=512):

    model_1 = models.create(args.arch,mb_h, num_features=args.features, dropout=args.dropout,
                            num_classes=ncs)
    #model_2 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)

    model_1_ema = models.create(args.arch,mb_h, num_features=args.features, dropout=args.dropout,
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

    initial_weights = load_checkpoint(args.init_1)
    copy_state_dict(initial_weights['state_dict'], model_1)
    copy_state_dict(initial_weights['state_dict'], model_1_ema)
    for i,cl in enumerate(ncs):
        exec('model_1_ema.module.classifier{}_{}.weight.data.copy_(model_1.module.classifier{}_{}.weight.data)'.format(i,cl,i,cl))

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
    def __init__(self, target_label, m, dis_gt, t_loader,N,hc=3, ncl=None,  n_epochs=200,
                 weight_decay=1e-5, ckpt_dir='/'):
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
        self.K_c=self.K
        self.model = m
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L = [torch.LongTensor(target_label[i]).to(self.dev) for i in range(len(self.K))]
        self.nmodel_gpus = 4#len()
        self.pseudo_loader = t_loader#torch.utils.data.DataLoader(t_loader,batch_size=256)
        # can also be DataLoader with less aug.
        self.train_loader = t_loader
        self.lamb = 25#args.lamb # the parameter lambda in the SK algorithm
        self.cpu=True
        self.dis_gt=dis_gt if dis_gt else None
        print('Using the distribution of source domain: {}'.format(self.dis_gt!=None))#target
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
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def write_sta_im(train_loader):
    label2num=collections.defaultdict(int)
    for x in train_loader:
        label2num[x[1]]+=1
    labels=sorted(label2num.items(),key=lambda item:item[1])[::-1]
    num = [j for i, j in labels]
    distribution = np.array(num)/len(train_loader)

    # x = [i for i in range(2,1041)]
    # x = np.array(x)
    # print('x is :\n', x)
    # num = [j for i,j in labels[2:]]
    # y = np.array(num)
    # print('y is :\n', y)
    # plt.plot(x, y, 'b-')
    # popt, pcov = curve_fit(func, x, y)
    # # popt数组中，三个值分别是待求参数a,b,c
    # y2 = [func(i, popt[0], popt[1], popt[2]) for i in x]
    # plt.plot(x, y2, 'r--')
    # plt.savefig("sta_labelnum.jpg")
    return num

def print_cluster_acc(label_dict,target_label):
    num_correct = 0
    for pid in label_dict:
        pid_index = np.asarray(label_dict[pid])
        pred_label = np.argmax(np.bincount(target_label[pid_index]))
        num_correct += (target_label[pid_index] == pred_label).astype(np.float32).sum()
    cluster_accuracy = num_correct / len(target_label)
    print(f'cluster accucary: {cluster_accuracy:.3f}')

def kmeans_cluster():
    pass

def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log{}.txt'.format(args.cluster_num)))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters > 0) else None

    ncs = [int(x) for x in args.ncs.split(',')]
    dataset_target, label_dict = get_data(args.dataset_target, args.data_dir, len(ncs))

    dataset_source, _ = get_data(args.dataset_source, args.data_dir, len(ncs))

    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    # Create model
    mb_h=2048

    model_1, _, model_1_ema, _ = create_model(args,ncs,mb_h)
    ecn=None
    # Evaluator


    evaluator_1 = Evaluator(model_1)
    evaluator_1_ema = Evaluator(model_1_ema)
    #evaluator_2_ema = Evaluator(model_2_ema)
    # evaluator_1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
    #                          cmc_flag=True)

    #target_label = np.load("target_label.npy")

    clusters = [args.num_clusters] * args.epochs# TODO: dropout clusters
    cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,
                                     testset=dataset_target.train)


    feature_length = args.features if args.features > 0 else 2048

    #o.L[0]=torch.LongTensor(target_label).cuda()
    target_label = []
    extracting_flag=False
    for nc_i in ncs:
        style='gem'#gem_augmix
        if args.cluster_num==0:
            plabel_path="pesudo_label/{}_{}/target_label{}{}.npy".format(args.arch,args.dataset_target,nc_i,style)
        else:
            plabel_path=os.path.join(args.pesudolabel_path,'target_label{}_{}.npy'.format(nc_i,args.cluster_num))
        if os.path.exists(plabel_path):
            target_label_tmp=np.load(plabel_path)
            # import random
            # target_label_tmp=np.array([random.randint(0, 699) for _ in range(len(dataset_target.train))],np.int32)
            print('\n {} existing\n'.format(plabel_path))
        else:
            if not extracting_flag:
                dict_f, _ = extract_features(model_1, cluster_loader, args.choice_c, print_freq=args.print_freq)
                moving_avg_features = torch.stack(list(dict_f.values())).numpy()
                extracting_flag=True
            print('\n Clustering into {} classes \n'.format(nc_i))
            print("Loading into {}".format(plabel_path))
            km = KMeans(n_clusters=nc_i, random_state=args.seed, n_jobs=args.n_jobs).fit(moving_avg_features)
            target_label_tmp = np.asarray(km.labels_)
            cluster_centers = np.asarray(km.cluster_centers_)

            np.save("pesudo_label/{}_{}/target_label{}{}.npy".format(args.arch,args.dataset_target,nc_i,style),target_label_tmp)
            # np.save("pesudo_label/{}/cluster_centers{}{}.npy".format(args.arch,nc_i,style),cluster_centers)

        target_label.append(target_label_tmp)

        print_cluster_acc(label_dict, target_label_tmp)
    #luster_centers = np.load("cluster_centers"+str(args.num_clusters)+".npy")
    # cluster_centers = np.load("cluster_centers700.npy")
    # model_1.module.classifier700.weight.data.copy_(torch.from_numpy(normalize(cluster_centers, axis=1)).float().cuda())

    distribution = write_sta_im(dataset_source.train)

    contrast = MemoryMoCo_id(mb_h, len(dataset_target.train), K = 4196, index2label=target_label, choice_c=args.choice_c,T=0.07,use_softmax=True, cluster_num=args.cluster_num).cuda()

    # ecn = InvNet(num_features=2048, num_classes=len(dataset_target.train), target_label=target_label,ncl=ncs,
    #              beta=0.05, knn=6, alpha=0.01).cuda()
    o =  Optimizer(target_label,dis_gt=distribution, m=model_1, ncl=ncs, t_loader=cluster_loader,N=len(dataset_target.train))

    num_opts=[4,2,1]
    # o.L[0] = torch.LongTensor(target_label).cuda()
    for epoch in range(len(clusters)):
        #
        # if epoch<=5:
        #     num_opt = num_opts[0]
        # elif 5<epoch<=25:
        #     num_opt = num_opts[1]
        # else:
        #     num_opt = num_opts[2]
        # for i in range(num_opt):
        #     o.optimize_labels()

        if epoch == args.epochs-1:
            prenc_i=-1
            dict_f, _ = extract_features(model_1_ema, cluster_loader, args.choice_c, print_freq=args.print_freq)
            moving_avg_features = torch.stack(list(dict_f.values())).numpy()
            for in_,nc_i in enumerate(ncs):
                if prenc_i==nc_i:
                    continue
                # if nc == 0 and os.path.exists("target_label"+str(nc)+"gem.npy"):
                #     continue
                print('\n Clustering into {} classes \n'.format(nc_i))
                km = KMeans(n_clusters=nc_i, random_state=args.seed, n_jobs=args.n_jobs).fit(moving_avg_features)
                target_label_tmp= np.asarray(km.labels_)
                cluster_centers = np.asarray(km.cluster_centers_)
                print("\n Loading into npy with {} fine-cluster\n".format(nc_i))


                np.save("{}/target_label{}_{}.npy".format(args.pesudolabel_path,nc_i,args.cluster_num+1),target_label_tmp)
                np.save("{}/cluster_centers{}_{}.npy".format(args.logs_dir,nc_i,args.cluster_num+1),cluster_centers)
                # o.L[in_]=target_label_tmp

                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                o.L[in_] = torch.LongTensor(target_label_tmp).to(dev)

                print_cluster_acc(label_dict, target_label_tmp)
                prenc_i=nc_i
                del km
            break
            # del model_1_ema
            # del model_1
            # model_1, _, model_1_ema, _ = create_model(args, ncs)
        # km = MiniBatchKMeans(n_clusters=clusters[nc], random_state=args.seed, n_jobs=8).fit(moving_avg_features)
        # print acc of kmeans

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
        for i in range(len(dataset_target.train)):
            dataset_target.train[i] = list(dataset_target.train[i])
            for j in range(len(ncs)):
                dataset_target.train[i][j+1] = int(target_label[j][i])
            dataset_target.train[i] = tuple(dataset_target.train[i])



        iters_=400 if epoch==0 else iters
        # print(nc,"============"+str(iters_))
        # cc=args.choice_c#(args.choice_c+1)%
        train_loader_target= get_train_loader(dataset_target, args.height, args.width, args.choice_c,
                                                         args.batch_size, args.workers, args.num_instances, iters_)

        # Optimizer
        params = []
        flag = 1.0
        # if 40<epoch<=80 or 120<epoch<=160 or 200<epoch:
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
        trainer = mltTrainer_single(model_1, model_1_ema, contrast, None,
                             num_cluster=ncs, alpha=args.alpha)


        train_loader_target.new_epoch()


        # index2label = dict([(i, j) for i, j in enumerate(np.asarray(target_label[0]))])
        # index2label1= dict([(i, j) for i, j in enumerate(np.asarray(target_label[1]))])
        # index2label2 = dict([(i, j) for i, j in enumerate(np.asarray(target_label[2]))])
        # contrast.index_pl=3-int(epoch/10)

        trainer.train(epoch, train_loader_target, optimizer, args.choice_c,
                      ce_soft_weight=args.soft_ce_weight, tri_soft_weight=args.soft_tri_weight,
                      print_freq=args.print_freq, train_iters=iters_)
        o.optimize_labels()


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
        if epoch==20:
            args.eval_step=5
        elif epoch==80:
            args.eval_step=1
        if ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            mAP_1 = evaluator_1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                                             cmc_flag=False)

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
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-st', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--choice_c', type=int, default=0)
    parser.add_argument('--num-clusters', type=int, default=700)
    parser.add_argument('--ncs', type=str, default='700')
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
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=200)
    # training configs


    parser.add_argument('--init-1', type=str, default='logs/market1501TOdukemtmc/resnet50-pretrain-1/model_best.pth.tar', metavar='PATH')
    parser.add_argument('--pesudolabel-path', type=str, default='', metavar='PATH')
    parser.add_argument('--cluster-num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--n-jobs', type=int, default=8)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs_moco'))
    print("======mlt_train_self-labeling=======")

    main()
