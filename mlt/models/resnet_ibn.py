from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from .gem_pooling import GeneralizedMeanPoolingP
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a


__all__ = ['ResNetIBN', 'resnet_ibn50a', 'resnet_ibn101a']


class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=None):
        super(ResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        print("GeneralizedMeanPoolingP")
        self.gap = GeneralizedMeanPoolingP(3)


        # self.memorybank_fc=nn.Sequential(
        #     nn.Linear(2048,512,bias=True),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 128,bias=False),
        #     nn.BatchNorm1d(128)
        # )
        #
        self.memorybank_fc = nn.Linear(2048, 2048)
        self.mbn = nn.BatchNorm1d(2048)
        init.kaiming_normal_(self.memorybank_fc.weight, mode='fan_out')
        init.constant_(self.memorybank_fc.bias, 0)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes is not None:
                for i,num_cluster in enumerate(num_classes):
                    exec("self.classifier{}_{} = nn.Linear(self.num_features, {}, bias=False)".format(i,num_cluster,num_cluster))
                    exec("init.normal_(self.classifier{}_{}.weight, std=0.001)".format(i,num_cluster))


        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, feature_withbn=False, training=False, cluster=False):
        x = self.base(x)

        x = self.gap(x)

        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)#1

        if training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:#FALSE
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:#FALSE
            bn_x = F.relu(bn_x)

        if self.dropout > 0:#FALSE
            bn_x = self.drop(bn_x)

        # if self.num_classes > 0:
        #     prob700 = self.classifier700(bn_x)#True, the number of cluster
        #     prob2000 = self.classifier2000(bn_x)
        #     prob3000 = self.classifier3000(bn_x)
        prob=[]
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("prob.append(self.classifier{}_{}(bn_x))".format(i,num_cluster))

        else:
            return x, bn_x

        #if feature_withbn:#False
        #    return bn_x, prob

        mb_x = self.mbn(self.memorybank_fc(bn_x))

        return x, prob, mb_x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNetIBN.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.relu.state_dict())
        self.base[3].load_state_dict(resnet.maxpool.state_dict())
        self.base[4].load_state_dict(resnet.layer1.state_dict())
        self.base[5].load_state_dict(resnet.layer2.state_dict())
        self.base[6].load_state_dict(resnet.layer3.state_dict())
        self.base[7].load_state_dict(resnet.layer4.state_dict())


def resnet_ibn50a(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNetIBN('101a', **kwargs)
