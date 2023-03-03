from __future__ import print_function, division
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os
from torch.optim import Adam
from torchvision import models
import torch.nn as tnn
from torch.nn import functional as F
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16(tnn.Module):
    def __init__(self, n_classes=3):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(8 * 8 * 512, 512)
        self.layer7 = vgg_fc_layer(512, 128)

        # Final layer
        self.layer8 = tnn.Linear(128, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        vgg16_features = self.layer6(out)
        out = self.layer7(vgg16_features)
        out = self.layer8(out)

        return vgg16_features, out


class VGG16_C(tnn.Module):
    def __init__(self, num_classes=2):
        super(VGG16_C, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([6, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(8 * 8 * 512, 512)
        self.layer7 = vgg_fc_layer(512, 128)

        # Final layer
        self.layer8 = tnn.Linear(128, num_classes)

    def forward(self, x, y):
        out = self.layer1(torch.cat([x, y],1))
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # print("-----------------------------")
        # print(out.shape)
        # print("-----------------------------")
        out = out.view(out.size(0), -1)

        vgg16_features = self.layer6(out)
        out = self.layer7(vgg16_features)
        out = self.layer8(out)

        return vgg16_features, out


class VGG16_C2(tnn.Module):   # Abs()
    def __init__(self, num_classes=2):
        super(VGG16_C2, self).__init__()
        self.adc = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(8 * 8 * 512, 512)
        self.layer7 = vgg_fc_layer(512, 128)

        # Final layer
        self.layer8 = tnn.Linear(128, num_classes)

    def forward(self, x, y):
        out_x = self.layer1(x)
        out_x = self.layer2(out_x)
        out_x = self.layer3(out_x)
        out_x = self.layer4(out_x)
        out_x = self.layer5(out_x)

        out_y = self.layer1(y)
        out_y = self.layer2(out_y)
        out_y = self.layer3(out_y)
        out_y = self.layer4(out_y)
        out_y = self.layer5(out_y)
        vgg16_features = self.adc(abs(out_x-out_y))
        # print(out.shape)
        out = vgg16_features.view(vgg16_features.size(0), -1)
        # print(out.shape)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out

if __name__ == '__main__':
    c=VGG16()
    cc = VGG16_C()
    print()
