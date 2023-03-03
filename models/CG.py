import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision import models
from .vit import ViT
from .swin import SwinTransformer
import functools


class GeneratorC(nn.Module):
    def __init__(self):
        super(GeneratorC, self).__init__()
        decoders = list(models.vgg16(pretrained=True).features.children())
        self.dec1 = nn.Sequential(*decoders[:5])
        self.dec2 = nn.Sequential(*decoders[5:10])
        self.dec3 = nn.Sequential(*decoders[10:17])
        self.dec4 = nn.Sequential(*decoders[17:24])
        self.dec5 = nn.Sequential(*decoders[24:])
        self.AD5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 5, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.AD4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 5, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.AD3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 5, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.AD2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 5, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.AD1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pred = nn.Conv2d(64, 1, 3)
        self.apply(self.init_model)
    def forward(self, x, y):
        x_f1 = self.dec1(x)
        x_f2 = self.dec1(x_f1)
        x_f3 = self.dec1(x_f2)
        x_f4 = self.dec1(x_f3)
        x_f5 = self.dec1(x_f4)

        y_f1 = self.dec1(y)
        y_f2 = self.dec1(y_f1)
        y_f3 = self.dec1(y_f2)
        y_f4 = self.dec1(y_f3)
        y_f5 = self.dec1(y_f4)

        ad_f5 = self.AD5(abs(x_f5 - y_f5))
        ad_f4 = self.AD4(torch.cat([abs(x_f4 - y_f4), ad_f5], 1))
        ad_f3 = self.AD3(torch.cat([abs(x_f3 - y_f3), ad_f4], 1))
        ad_f2 = self.AD2(torch.cat([abs(x_f2 - y_f2), ad_f3], 1))
        ad_f1 = self.AD1(torch.cat([abs(x_f1 - y_f1), ad_f2], 1))
        out = self.pred(ad_f1)
        return out

    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True


class DiscriminatorC(nn.Module):
    def __init__(self):
        super(DiscriminatorC, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.features[0] = nn.Conv2d(6, 64, 3, 1, 1)
        self.model.classifier = nn.Sequential(
            nn.Conv2d(512, 512, 7),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
        )
        self.apply(self.init_model)

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))
    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True


class GeneratorVGG(nn.Module):
    def __init__(self):
        super(GeneratorVGG, self).__init__()
        model = models.vgg16(pretrained=True)
        # model.features[0] = nn.Conv2d(6, 64, 3, 1, 1)
        decoders = list(model.features.children())
        self.enc1 = nn.Sequential(*decoders[:5])
        self.enc2 = nn.Sequential(*decoders[5:10])
        self.enc3 = nn.Sequential(*decoders[10:17])
        self.enc4 = nn.Sequential(*decoders[17:24])
        self.enc5 = nn.Sequential(*decoders[24:])

        self.dec5 = nn.Sequential(
            # nn.Upsample(scale_factor=scale_factor, mode='bilinear')
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(1024, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            nn.Conv2d(128, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.mask = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.AD5 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            # nn.Conv2d(512, 512, 1, 1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.AD4 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            # nn.Conv2d(256, 256, 1, 1, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Conv2d(768, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.AD3 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            # nn.Conv2d(256, 256, 1, 1, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Conv2d(384, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.AD2 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            # nn.Conv2d(128, 128, 1, 1, 1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            nn.Conv2d(192, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.AD1 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            # nn.Conv2d(64, 64, 1, 1, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.pred = nn.Conv2d(64, 1, 3)
        self.apply(self.init_model)
    def forward(self, x, y):
        x_f1 = self.enc1(x)  # 64
        x_f2 = self.enc2(x_f1)  # 128
        x_f3 = self.enc3(x_f2)  # 256
        x_f4 = self.enc4(x_f3)  # 512
        x_f5 = self.enc5(x_f4)  # 512
        # print("------------------------------")
        # print("x1.shape:{}".format(x_f1.shape))
        # print("x2.shape:{}".format(x_f2.shape))
        # print("x3.shape:{}".format(x_f3.shape))
        # print("x4.shape:{}".format(x_f4.shape))
        # print("x5.shape:{}".format(x_f5.shape))
        # print("------------------------------")

        y_f1 = self.enc1(y)  # 64
        y_f2 = self.enc2(y_f1)  # 128
        y_f3 = self.enc3(y_f2)  # 256
        y_f4 = self.enc4(y_f3)  # 512
        y_f5 = self.enc5(y_f4)  # 512

        #-----------------------------
        x_dec5 = self.dec5(x_f5)

        x_dec4 = self.dec4(torch.cat([x_dec5, x_f4], 1))
        # print(x_dec4.shape)
        x_dec3 = self.dec3(torch.cat([x_dec4, x_f3], 1))
        x_dec2 = self.dec2(torch.cat([x_dec3, x_f2], 1))
        # print(x_dec2.shape)
        x_dec1 = self.dec1(torch.cat([x_dec2, x_f1], 1))

        y_dec5 = self.dec5(y_f5)
        y_dec4 = self.dec4(torch.cat([y_dec5, y_f4], 1))
        y_dec3 = self.dec3(torch.cat([y_dec4, y_f3], 1))
        y_dec2 = self.dec2(torch.cat([y_dec3, y_f2], 1))
        y_dec1 = self.dec1(torch.cat([y_dec2, y_f1], 1))
        #-----------------------------

        ad5 = self.AD5(abs(x_dec5 - y_dec5))
        ad4 = self.AD4(torch.cat([F.upsample_bilinear(ad5, scale_factor=2), abs(x_dec4 - y_dec4)], 1))
        ad3 = self.AD3(torch.cat([F.upsample_bilinear(ad4, scale_factor=2), abs(x_dec3 - y_dec3)], 1))
        ad2 = self.AD2(torch.cat([F.upsample_bilinear(ad3, scale_factor=2), abs(x_dec2 - y_dec2)], 1))
        ad1 = self.AD1(torch.cat([ad2, abs(x_dec1 - y_dec1)], 1))

        out = F.upsample_bilinear(self.mask(ad1), scale_factor=2)
        # print(out.shape)

        return out

    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True



class GeneratorVGGC(nn.Module):
    def __init__(self):
        super(GeneratorVGGC, self).__init__()
        model = models.vgg16(pretrained=True)
        # model.features[0] = nn.Conv2d(6, 64, 3, 1, 1)
        decoders = list(model.features.children())
        self.enc1 = nn.Sequential(*decoders[:5])
        self.enc2 = nn.Sequential(*decoders[5:10])
        self.enc3 = nn.Sequential(*decoders[10:17])
        self.enc4 = nn.Sequential(*decoders[17:24])
        self.enc5 = nn.Sequential(*decoders[24:])

        self.dec5 = nn.Sequential(
            # nn.Upsample(scale_factor=scale_factor, mode='bilinear')
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(1024, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            nn.Conv2d(128, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.mask = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.AD5 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            nn.Conv2d(1024, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.AD4 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            nn.Conv2d(1024, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.AD3 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            nn.Conv2d(512, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.AD2 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            nn.Conv2d(256, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.AD1 = nn.Sequential(
            # nn.UpsamplingBilinear2d(2),
            nn.Conv2d(128, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.pred = nn.Conv2d(64, 1, 3)
        self.apply(self.init_model)
    def forward(self, x, y):
        x_f1 = self.enc1(x)  # 64
        x_f2 = self.enc2(x_f1)  # 128
        x_f3 = self.enc3(x_f2)  # 256
        x_f4 = self.enc4(x_f3)  # 512
        x_f5 = self.enc5(x_f4)  # 512
        # print("------------------------------")
        # print("x1.shape:{}".format(x_f1.shape))
        # print("x2.shape:{}".format(x_f2.shape))
        # print("x3.shape:{}".format(x_f3.shape))
        # print("x4.shape:{}".format(x_f4.shape))
        # print("x5.shape:{}".format(x_f5.shape))
        # print("------------------------------")

        y_f1 = self.enc1(y)  # 64
        y_f2 = self.enc2(y_f1)  # 128
        y_f3 = self.enc3(y_f2)  # 256
        y_f4 = self.enc4(y_f3)  # 512
        y_f5 = self.enc5(y_f4)  # 512

        #-----------------------------
        x_dec5 = self.dec5(x_f5)

        x_dec4 = self.dec4(torch.cat([x_dec5, x_f4], 1))
        # print(x_dec4.shape)
        x_dec3 = self.dec3(torch.cat([x_dec4, x_f3], 1))
        x_dec2 = self.dec2(torch.cat([x_dec3, x_f2], 1))
        # print(x_dec2.shape)
        x_dec1 = self.dec1(torch.cat([x_dec2, x_f1], 1))

        y_dec5 = self.dec5(y_f5)
        y_dec4 = self.dec4(torch.cat([y_dec5, y_f4], 1))
        y_dec3 = self.dec3(torch.cat([y_dec4, y_f3], 1))
        y_dec2 = self.dec2(torch.cat([y_dec3, y_f2], 1))
        y_dec1 = self.dec1(torch.cat([y_dec2, y_f1], 1))
        #-----------------------------

        ad5 = self.AD5(torch.cat([x_dec5, y_dec5], 1))
        ad4 = self.AD4(torch.cat([F.upsample_bilinear(ad5, scale_factor=2), x_dec4, y_dec4], 1))
        ad3 = self.AD3(torch.cat([F.upsample_bilinear(ad4, scale_factor=2), x_dec3, y_dec3], 1))
        ad2 = self.AD2(torch.cat([F.upsample_bilinear(ad3, scale_factor=2), x_dec2, y_dec2], 1))
        ad1 = self.AD1(torch.cat([ad2, x_dec1, y_dec1], 1))

        out = F.upsample_bilinear(self.mask(ad1), scale_factor=2)
        # print(out.shape)

        return out

    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True


class DiscriminatorCU(nn.Module):
    def __init__(self):
        super(DiscriminatorCU, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.features[0] = nn.Conv2d(6, 64, 3, 1, 1)
        self.model.classifier = nn.Sequential(
            # nn.Conv2d(512, 512, 7),
            nn.Linear(512*7*7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
        )
        self.apply(self.init_model)

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))
    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True


class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class EncoderAD(nn.Module):
    def __init__(self):
        super(EncoderAD, self).__init__()
        encoders = list(models.vgg16(pretrained=True).features.children())
        self.enc1 = nn.Sequential(*encoders[:5])
        self.enc2 = nn.Sequential(*encoders[5:10])
        self.enc3 = nn.Sequential(*encoders[10:17])
        self.enc4 = nn.Sequential(*encoders[17:24])
        self.enc5 = nn.Sequential(*encoders[24:])
        # self.AD5 = nn.Sequential(
        #     nn.Conv2d(512, 512, 3, 1, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )

        # self.AD4 = nn.Sequential(
        #     nn.Conv2d(512, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        # self.AD3 = nn.Sequential(
        #     nn.Conv2d(256, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )
        # self.AD2 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )
        # self.AD1 = nn.Sequential(
        #     nn.Conv2d(64, 32, 3, 1, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True)
        # )
        self.AD5 = SegNetEnc(512,512,2,1)
        self.AD4 = SegNetEnc(512,256,2,1)
        self.AD3 = SegNetEnc(256,128,2,1)
        self.AD2 = SegNetEnc(128,64,2,1)
        self.AD1 = SegNetEnc(64,32,2,1)
        self.apply(self.init_model)
    def forward(self, x, y):
        x_f1 = self.enc1(x)
        x_f2 = self.enc2(x_f1)
        x_f3 = self.enc3(x_f2)
        x_f4 = self.enc4(x_f3)
        x_f5 = self.enc5(x_f4)

        y_f1 = self.enc1(y)
        y_f2 = self.enc2(y_f1)
        y_f3 = self.enc3(y_f2)
        y_f4 = self.enc4(y_f3)
        y_f5 = self.enc5(y_f4)

        ad_f5 = self.AD5(abs(x_f5 - y_f5))
        ad_f4 = self.AD4(abs(x_f4 - y_f4))
        ad_f3 = self.AD3(abs(x_f3 - y_f3))
        ad_f2 = self.AD2(abs(x_f2 - y_f2))
        ad_f1 = self.AD1(abs(x_f1 - y_f1))
        
        return ad_f1, ad_f2, ad_f3, ad_f4, ad_f5

    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True



class DecoderAD(nn.Module): 
    def __init__(self):
        super(DecoderAD, self).__init__()

        self.GC5 = SegNetEnc(512,512,1,1)
        self.GC4 = SegNetEnc(768,256,1,1)
        self.GC3 = SegNetEnc(384,128,1,1)
        self.GC2 = SegNetEnc(192,64,1,1)
        self.GC1 = SegNetEnc(96,32,1,1)
        self.predection = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        # self.apply(self.init_model)
    def forward(self, ad1, ad2, ad3, ad4, ad5):

        # 512 256 128 64 32
        ad5 = self.GC5(ad5)
        ad4 = self.GC4(torch.cat([F.upsample_bilinear(ad5, scale_factor=2), ad4], 1))
        ad3 = self.GC3(torch.cat([F.upsample_bilinear(ad4, scale_factor=2), ad3], 1))
        ad2 = self.GC2(torch.cat([F.upsample_bilinear(ad3, scale_factor=2), ad2], 1))
        ad1 = self.GC1(torch.cat([F.upsample_bilinear(ad2, scale_factor=2), ad1], 1))
        out = self.predection(ad1)
        return ad1, out

  

class EncoderAD3(nn.Module):
    def __init__(self):
        super(EncoderAD3, self).__init__()
        encoders = list(models.vgg16(pretrained=True).features.children())
        self.enc1 = nn.Sequential(*encoders[:5])
        self.enc2 = nn.Sequential(*encoders[5:10])
        self.enc3 = nn.Sequential(*encoders[10:17])
        self.enc4 = nn.Sequential(*encoders[17:24])
        self.enc5 = nn.Sequential(*encoders[24:])

        self.AD5 = SegNetEnc(512,512,2,1)
        self.AD4 = SegNetEnc(512,256,2,1)
        self.AD3 = SegNetEnc(256,128,2,1)
        self.AD2 = SegNetEnc(128,64,2,1)
        self.AD1 = SegNetEnc(64,64,2,1)
        self.apply(self.init_model)
    def forward(self, x, y):
        x_f1 = self.enc1(x)
        x_f2 = self.enc2(x_f1)
        x_f3 = self.enc3(x_f2)
        x_f4 = self.enc4(x_f3)
        x_f5 = self.enc5(x_f4)

        y_f1 = self.enc1(y)
        y_f2 = self.enc2(y_f1)
        y_f3 = self.enc3(y_f2)
        y_f4 = self.enc4(y_f3)
        y_f5 = self.enc5(y_f4)

        ad_f5 = self.AD5(abs(x_f5 - y_f5))
        ad_f4 = self.AD4(abs(x_f4 - y_f4))
        ad_f3 = self.AD3(abs(x_f3 - y_f3))
        ad_f2 = self.AD2(abs(x_f2 - y_f2))
        ad_f1 = self.AD1(abs(x_f1 - y_f1))
        
        return ad_f1, ad_f2, ad_f3, ad_f4, ad_f5

    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True



class DecoderAD3(nn.Module): 
    def __init__(self):
        super(DecoderAD3, self).__init__()

        self.GC5 = SegNetEnc(512,512,1,1)
        self.GC4 = SegNetEnc(768,256,1,1)
        self.GC3 = SegNetEnc(384,128,1,1)
        self.GC2 = SegNetEnc(192,64,1,1)
        self.GC1 = SegNetEnc(128,64,1,1)
        self.predection = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        # self.apply(self.init_model)
    def forward(self, ad1, ad2, ad3, ad4, ad5):

        # 512 256 128 64 32
        ad5 = self.GC5(ad5)
        ad4 = self.GC4(torch.cat([F.upsample_bilinear(ad5, scale_factor=2), ad4], 1))
        ad3 = self.GC3(torch.cat([F.upsample_bilinear(ad4, scale_factor=2), ad3], 1))
        ad2 = self.GC2(torch.cat([F.upsample_bilinear(ad3, scale_factor=2), ad2], 1))
        ad1 = self.GC1(torch.cat([F.upsample_bilinear(ad2, scale_factor=2), ad1], 1))
        out = self.predection(ad1)
        return ad1, out

  
  

class EncoderAD4(nn.Module):
    def __init__(self):
        super(EncoderAD4, self).__init__()
        encoders = list(models.vgg16(pretrained=True).features.children())
        self.enc1 = nn.Sequential(*encoders[:5])
        self.enc2 = nn.Sequential(*encoders[5:10])
        self.enc3 = nn.Sequential(*encoders[10:17])
        self.enc4 = nn.Sequential(*encoders[17:24])
        self.enc5 = nn.Sequential(*encoders[24:])

        self.AD5 = SegNetEnc(512,256,2,1)
        self.AD4 = SegNetEnc(512,128,2,1)
        self.AD3 = SegNetEnc(256,128,2,1)
        self.AD2 = SegNetEnc(128,64,2,1)
        self.AD1 = SegNetEnc(64,32,2,1)
        self.apply(self.init_model)
    def forward(self, x, y):
        x_f1 = self.enc1(x)
        x_f2 = self.enc2(x_f1)
        x_f3 = self.enc3(x_f2)
        x_f4 = self.enc4(x_f3)
        x_f5 = self.enc5(x_f4)

        y_f1 = self.enc1(y)
        y_f2 = self.enc2(y_f1)
        y_f3 = self.enc3(y_f2)
        y_f4 = self.enc4(y_f3)
        y_f5 = self.enc5(y_f4)

        ad_f5 = self.AD5(abs(x_f5 - y_f5))
        ad_f4 = self.AD4(abs(x_f4 - y_f4))
        ad_f3 = self.AD3(abs(x_f3 - y_f3))
        ad_f2 = self.AD2(abs(x_f2 - y_f2))
        ad_f1 = self.AD1(abs(x_f1 - y_f1))
        
        return ad_f1, ad_f2, ad_f3, ad_f4, ad_f5

    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True



class DecoderAD4(nn.Module): 
    def __init__(self):
        super(DecoderAD4, self).__init__()

        # self.GC5 = SegNetEnc(256,256,1,1)
        self.GC4 = SegNetEnc(768,256,2,1)
        self.GC3 = SegNetEnc(1024,128,2,1)
        self.GC2 = SegNetEnc(896,64,2,1)
        self.GC1 = SegNetEnc(832,32,2,1)
        self.predection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        # self.apply(self.init_model)
    def forward(self, ad1, ad2, ad3, ad4):

        # 512 256 128 64 32
        ad4 = self.GC4(ad4)
        ad3 = self.GC3(torch.cat([ad4, ad3], 1))
        ad2 = self.GC2(torch.cat([ad3, ad2], 1))
        ad1 = self.GC1(torch.cat([ad2, ad1], 1))
        # ad1 = self.GC1(torch.cat([F.upsample_bilinear(ad2, scale_factor=2), ad1], 1))
        out = self.predection(ad1)
        return ad1, out

  

class DecoderAD4_v2(nn.Module): 
    def __init__(self):
        super(DecoderAD4_v2, self).__init__()

        # self.GC5 = SegNetEnc(256,256,1,1)
        self.GC4 = SegNetEnc(768,256,2,1)
        self.GC3 = SegNetEnc(640,128,2,1)
        self.GC2 = SegNetEnc(320,64,2,1)
        self.GC1 = SegNetEnc(160,32,2,1)
        self.predection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        # self.apply(self.init_model)
    def forward(self, ad1, ad2, ad3, ad4):

        # 512 256 128 64 32
        ad4 = self.GC4(ad4)
        ad3 = self.GC3(torch.cat([ad4, ad3], 1))
        ad2 = self.GC2(torch.cat([ad3, ad2], 1))
        ad1 = self.GC1(torch.cat([ad2, ad1], 1))
        # ad1 = self.GC1(torch.cat([F.upsample_bilinear(ad2, scale_factor=2), ad1], 1))
        out = self.predection(ad1)
        return ad1, out

  

class EncoderFusion(nn.Module):
    def __init__(self):
        super(EncoderFusion, self).__init__()
        encoders = list(models.vgg16(pretrained=True).features.children())
        self.enc1 = nn.Sequential(*encoders[:5])
        self.enc2 = nn.Sequential(*encoders[5:10])
        self.enc3 = nn.Sequential(*encoders[10:17])
        self.enc4 = nn.Sequential(*encoders[17:24])
        self.enc5 = nn.Sequential(*encoders[24:])

        self.AD5 = SegNetEnc(1024,512,2,1)
        self.AD4 = SegNetEnc(1024,256,2,1)
        self.AD3 = SegNetEnc(512,128,2,1)
        self.AD2 = SegNetEnc(256,64,2,1)
        self.AD1 = SegNetEnc(128,32,2,1)
        self.apply(self.init_model)
    def forward(self, x, y):
        x_f1 = self.enc1(x)
        x_f2 = self.enc2(x_f1)
        x_f3 = self.enc3(x_f2)
        x_f4 = self.enc4(x_f3)
        x_f5 = self.enc5(x_f4)

        y_f1 = self.enc1(y)
        y_f2 = self.enc2(y_f1)
        y_f3 = self.enc3(y_f2)
        y_f4 = self.enc4(y_f3)
        y_f5 = self.enc5(y_f4)

        ad_f5 = self.AD5(torch.cat([x_f5 , y_f5],1))
        ad_f4 = self.AD4(torch.cat([x_f4 , y_f4],1))
        ad_f3 = self.AD3(torch.cat([x_f3 , y_f3],1))
        ad_f2 = self.AD2(torch.cat([x_f2 , y_f2],1))
        ad_f1 = self.AD1(torch.cat([x_f1 , y_f1],1))
        
        return ad_f1, ad_f2, ad_f3, ad_f4, ad_f5

    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True



class DecoderFusion(nn.Module): 
    def __init__(self):
        super(DecoderFusion, self).__init__()

        self.GC5 = SegNetEnc(512,512,1,1)
        self.GC4 = SegNetEnc(768,256,1,1)
        self.GC3 = SegNetEnc(384,128,1,1)
        self.GC2 = SegNetEnc(192,64,1,1)
        self.GC1 = SegNetEnc(96,32,1,1)
        self.predection = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        # self.apply(self.init_model)
    def forward(self, ad1, ad2, ad3, ad4, ad5):

        # 512 256 128 64 32
        ad5 = self.GC5(ad5)
        ad4 = self.GC4(torch.cat([F.upsample_bilinear(ad5, scale_factor=2), ad4], 1))
        ad3 = self.GC3(torch.cat([F.upsample_bilinear(ad4, scale_factor=2), ad3], 1))
        ad2 = self.GC2(torch.cat([F.upsample_bilinear(ad3, scale_factor=2), ad2], 1))
        ad1 = self.GC1(torch.cat([F.upsample_bilinear(ad2, scale_factor=2), ad1], 1))
        out = self.predection(ad1)
        return ad1, out

  
class WSCD(nn.Module):
    def __init__(self):
        super(WSCD, self).__init__()
        self.encoder = EncoderAD()
        self.decoder = DecoderAD()
    def forward(self, C, UC, test=False):
        ad1, ad2, ad3, ad4, ad5 = self.encoder(C[0], C[1])
        cf, cmask = self.decoder(ad1, ad2, ad3, ad4, ad5)
        if test:
            return cmask
        uad1, uad2, uad3, uad4, uad5 = self.encoder(UC[0], UC[1])
        ucf, ucmask = self.decoder(uad1, uad2, uad3, uad4, uad5)

        # synthesis the change feature
        rcm1 = F.upsample_bilinear(cmask, ad1.size()[2:])
        rcm2 = F.upsample_bilinear(cmask, ad2.size()[2:])
        rcm3 = F.upsample_bilinear(cmask, ad3.size()[2:])
        rcm4 = F.upsample_bilinear(cmask, ad4.size()[2:])
        rcm5 = F.upsample_bilinear(cmask, ad5.size()[2:])
        
        # rucm1 = F.upsample_bilinear(ucmask, uad1.size()[2:])
        # rucm2 = F.upsample_bilinear(ucmask, uad2.size()[2:])
        # rucm3 = F.upsample_bilinear(ucmask, uad3.size()[2:])
        # rucm4 = F.upsample_bilinear(ucmask, uad4.size()[2:])
        # rucm5 = F.upsample_bilinear(ucmask, uad5.size()[2:])

        sc_ad1 = ad1 * rcm1 + uad1 * (1.-rcm1)
        sc_ad2 = ad2 * rcm2 + uad2 * (1.-rcm2)
        sc_ad3 = ad3 * rcm3 + uad3 * (1.-rcm3)
        sc_ad4 = ad4 * rcm4 + uad4 * (1.-rcm4)
        sc_ad5 = ad5 * rcm5 + uad5 * (1.-rcm5)

        suc_ad1 = uad1 * rcm1 + ad1 * (1.-rcm1)
        suc_ad2 = uad2 * rcm2 + ad2 * (1.-rcm2)
        suc_ad3 = uad3 * rcm3 + ad3 * (1.-rcm3)
        suc_ad4 = uad4 * rcm4 + ad4 * (1.-rcm4)
        suc_ad5 = uad5 * rcm5 + ad5 * (1.-rcm5)

        scf, scmask  = self.decoder(sc_ad1, sc_ad2, sc_ad3, sc_ad4, sc_ad5)
        sucf, sucmask = self.decoder(suc_ad1, suc_ad2, suc_ad3, suc_ad4, suc_ad5)

        return cmask, ucmask, scmask, sucmask, cf, ucf, scf, sucf

  
class WSCD2(nn.Module):
    def __init__(self):
        super(WSCD2, self).__init__()
        self.encoder = EncoderAD()
        self.decoder = DecoderAD()
    def forward(self, C, UC, test=False):
        ad1, ad2, ad3, ad4, ad5 = self.encoder(C[0], C[1])
        cf, cmask = self.decoder(ad1, ad2, ad3, ad4, ad5)
        if test:
            return cmask
        uad1, uad2, uad3, uad4, uad5 = self.encoder(UC[0], UC[1])
        ucf, ucmask = self.decoder(uad1, uad2, uad3, uad4, uad5)

        # synthesis the change feature
        # rcm1 = F.upsample_bilinear(cmask, ad1.size()[2:])
        # rcm2 = F.upsample_bilinear(cmask, ad2.size()[2:])
        # rcm3 = F.upsample_bilinear(cmask, ad3.size()[2:])
        # rcm4 = F.upsample_bilinear(cmask, ad4.size()[2:])
        # rcm5 = F.upsample_bilinear(cmask, ad5.size()[2:])
        


        sc_ad1 = ad1 + uad1
        sc_ad2 = ad2 + uad2
        sc_ad3 = ad3 + uad3
        sc_ad4 = ad4 + uad4
        sc_ad5 = ad5 + uad5



        scf, scmask  = self.decoder(sc_ad1, sc_ad2, sc_ad3, sc_ad4, sc_ad5)
        # sucf, sucmask = self.decoder(suc_ad1, suc_ad2, suc_ad3, suc_ad4, suc_ad5)

        return cmask, ucmask, scmask, cf, ucf, scf


  
class WSCD3(nn.Module):
    def __init__(self):
        super(WSCD3, self).__init__()
        self.encoder = EncoderAD3()
        self.decoder = DecoderAD3()
    def forward(self, C, UC, test=False):
        ad1, ad2, ad3, ad4, ad5 = self.encoder(C[0], C[1])
        cf, cmask = self.decoder(ad1, ad2, ad3, ad4, ad5)
        if test:
            return cmask
        uad1, uad2, uad3, uad4, uad5 = self.encoder(UC[0], UC[1])
        ucf, ucmask = self.decoder(uad1, uad2, uad3, uad4, uad5)

        # synthesis the change feature
        rcm1 = F.upsample_bilinear(cmask, ad1.size()[2:])
        rcm2 = F.upsample_bilinear(cmask, ad2.size()[2:])
        rcm3 = F.upsample_bilinear(cmask, ad3.size()[2:])
        rcm4 = F.upsample_bilinear(cmask, ad4.size()[2:])
        rcm5 = F.upsample_bilinear(cmask, ad5.size()[2:])
        


        sc_ad1 = ad1 + uad1
        sc_ad2 = ad2 + uad2
        sc_ad3 = ad3 + uad3
        sc_ad4 = ad4 + uad4
        sc_ad5 = ad5 + uad5



        scf, scmask  = self.decoder(sc_ad1, sc_ad2, sc_ad3, sc_ad4, sc_ad5)
        # sucf, sucmask = self.decoder(suc_ad1, suc_ad2, suc_ad3, suc_ad4, suc_ad5)

        return cmask, ucmask, scmask, cf, ucf, scf


  
class WSCD4(nn.Module):
    def __init__(self):
        super(WSCD4, self).__init__()
        self.encoder = EncoderAD4()
        self.decoder = DecoderAD4()
    def forward(self, C, UC, test=False):
        ad1, ad2, ad3, ad4, ad5 = self.encoder(C[0], C[1])
        cf, cmask = self.decoder(ad1, ad2, ad3, ad4, ad5)
        if test:
            return cmask
        uad1, uad2, uad3, uad4, uad5 = self.encoder(UC[0], UC[1])
        ucf, ucmask = self.decoder(uad1, uad2, uad3, uad4, uad5)

        # synthesis the change feature
        rcm1 = F.upsample_bilinear(cmask, ad1.size()[2:])
        rcm2 = F.upsample_bilinear(cmask, ad2.size()[2:])
        rcm3 = F.upsample_bilinear(cmask, ad3.size()[2:])
        rcm4 = F.upsample_bilinear(cmask, ad4.size()[2:])
        rcm5 = F.upsample_bilinear(cmask, ad5.size()[2:])
        
        sc_ad1 = ad1 + uad1
        sc_ad2 = ad2 + uad2
        sc_ad3 = ad3 + uad3
        sc_ad4 = ad4 + uad4
        sc_ad5 = ad5 + uad5

        scf, scmask  = self.decoder(sc_ad1, sc_ad2, sc_ad3, sc_ad4, sc_ad5)
        # sucf, sucmask = self.decoder(suc_ad1, suc_ad2, suc_ad3, suc_ad4, suc_ad5)

        return cmask, ucmask, scmask, cf, ucf, scf

  
class WSCD5(nn.Module):
    def __init__(self):
        super(WSCD5, self).__init__()
        self.encoder = EncoderFusion()
        self.decoder = DecoderFusion()
    def forward(self, C, UC, test=False):
        ad1, ad2, ad3, ad4, ad5 = self.encoder(C[0], C[1])
        cf, cmask = self.decoder(ad1, ad2, ad3, ad4, ad5)
        if test:
            return cmask
        uad1, uad2, uad3, uad4, uad5 = self.encoder(UC[0], UC[1])
        ucf, ucmask = self.decoder(uad1, uad2, uad3, uad4, uad5)

        # synthesis the change feature
        rcm1 = F.upsample_bilinear(cmask, ad1.size()[2:])
        rcm2 = F.upsample_bilinear(cmask, ad2.size()[2:])
        rcm3 = F.upsample_bilinear(cmask, ad3.size()[2:])
        rcm4 = F.upsample_bilinear(cmask, ad4.size()[2:])
        rcm5 = F.upsample_bilinear(cmask, ad5.size()[2:])
        
        sc_ad1 = ad1 + uad1
        sc_ad2 = ad2 + uad2
        sc_ad3 = ad3 + uad3
        sc_ad4 = ad4 + uad4
        sc_ad5 = ad5 + uad5

        scf, scmask  = self.decoder(sc_ad1, sc_ad2, sc_ad3, sc_ad4, sc_ad5)
        # sucf, sucmask = self.decoder(suc_ad1, suc_ad2, suc_ad3, suc_ad4, suc_ad5)

        return cmask, ucmask, scmask, cf, ucf, scf



  
class WSCD2_A1(nn.Module):
    def __init__(self):
        super(WSCD2_A1, self).__init__()
        self.encoder = EncoderAD()
        self.decoder = DecoderAD()
    def forward(self, C, UC, test=False):
        ad1, ad2, ad3, ad4, ad5 = self.encoder(C[0], C[1])
        cf, cmask = self.decoder(ad1, ad2, ad3, ad4, ad5)
        if test:
            return cmask
        uad1, uad2, uad3, uad4, uad5 = self.encoder(UC[0], UC[1])
        ucf, ucmask = self.decoder(uad1, uad2, uad3, uad4, uad5)

        # synthesis the change feature
        # rcm1 = F.upsample_bilinear(cmask, ad1.size()[2:])
        # rcm2 = F.upsample_bilinear(cmask, ad2.size()[2:])
        # rcm3 = F.upsample_bilinear(cmask, ad3.size()[2:])
        # rcm4 = F.upsample_bilinear(cmask, ad4.size()[2:])
        # rcm5 = F.upsample_bilinear(cmask, ad5.size()[2:])
        


        # sc_ad1 = ad1 + uad1
        # sc_ad2 = ad2 + uad2
        # sc_ad3 = ad3 + uad3
        # sc_ad4 = ad4 + uad4
        # sc_ad5 = ad5 + uad5



        # scf, scmask  = self.decoder(sc_ad1, sc_ad2, sc_ad3, sc_ad4, sc_ad5)
        # sucf, sucmask = self.decoder(suc_ad1, suc_ad2, suc_ad3, suc_ad4, suc_ad5)

        return cmask, ucmask, cf, ucf


  
class WSCD2_sub(nn.Module):
    def __init__(self):
        super(WSCD2_sub, self).__init__()
        self.encoder = EncoderAD()
        self.decoder = DecoderAD()
    def forward(self, C, UC, test=False):
        ad1, ad2, ad3, ad4, ad5 = self.encoder(C[0], C[1])
        cf, cmask = self.decoder(ad1, ad2, ad3, ad4, ad5)
        if test:
            return cmask
        uad1, uad2, uad3, uad4, uad5 = self.encoder(UC[0], UC[1])
        ucf, ucmask = self.decoder(uad1, uad2, uad3, uad4, uad5)

        # synthesis the change feature
        # rcm1 = F.upsample_bilinear(cmask, ad1.size()[2:])
        # rcm2 = F.upsample_bilinear(cmask, ad2.size()[2:])
        # rcm3 = F.upsample_bilinear(cmask, ad3.size()[2:])
        # rcm4 = F.upsample_bilinear(cmask, ad4.size()[2:])
        # rcm5 = F.upsample_bilinear(cmask, ad5.size()[2:])
        


        sc_ad1 = abs(ad1 - uad1)
        sc_ad2 = abs(ad2 - uad2)
        sc_ad3 = abs(ad3 - uad3)
        sc_ad4 = abs(ad4 - uad4)
        sc_ad5 = abs(ad5 - uad5)



        scf, scmask  = self.decoder(sc_ad1, sc_ad2, sc_ad3, sc_ad4, sc_ad5)
        # sucf, sucmask = self.decoder(suc_ad1, suc_ad2, suc_ad3, suc_ad4, suc_ad5)

        return cmask, ucmask, scmask, cf, ucf, scf


  
class WSCD2_ViT(nn.Module):
    def __init__(self):
        super(WSCD2_ViT, self).__init__()
        self.encoder = ViT()
        self.decoder = DecoderAD4()
    def forward(self, C, UC, test=False):
        # out = self.encoder(C[0])
        # print("---------------------------")
        # print(len(out))
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)
        # print(out[3].shape)
        # print("---------------------------")
        ad1, ad2, ad3, ad4 = self.encoder(C[0], C[1])
        cf, cmask = self.decoder(ad1, ad2, ad3, ad4)
        if test:
            return cmask
        uad1, uad2, uad3, uad4 = self.encoder(UC[0], UC[1])
        ucf, ucmask = self.decoder(uad1, uad2, uad3, uad4)

        # synthesis the change feature
        # rcm1 = F.upsample_bilinear(cmask, ad1.size()[2:])
        # rcm2 = F.upsample_bilinear(cmask, ad2.size()[2:])
        # rcm3 = F.upsample_bilinear(cmask, ad3.size()[2:])
        # rcm4 = F.upsample_bilinear(cmask, ad4.size()[2:])
        # rcm5 = F.upsample_bilinear(cmask, ad5.size()[2:])
        


        sc_ad1 = ad1 + uad1
        sc_ad2 = ad2 + uad2
        sc_ad3 = ad3 + uad3
        sc_ad4 = ad4 + uad4
        # sc_ad5 = ad5 + uad5



        scf, scmask  = self.decoder(sc_ad1, sc_ad2, sc_ad3, sc_ad4)
        # sucf, sucmask = self.decoder(suc_ad1, suc_ad2, suc_ad3, suc_ad4, suc_ad5)

        return cmask, ucmask, scmask, cf, ucf, scf

  
class WSCD2_Swin(nn.Module):
    def __init__(self):
        super(WSCD2_Swin, self).__init__()
        self.encoder = SwinTransformer()
        self.decoder = DecoderAD4_v2()
    def forward(self, C, UC, test=False):
        # out = self.encoder(C[0])
        # print("---------------------------")
        # print(len(out))
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)
        # print(out[3].shape)
        # print("---------------------------")
        ad1, ad2, ad3, ad4 = self.encoder(C[0], C[1])
        cf, cmask = self.decoder(ad1, ad2, ad3, ad4)
        if test:
            return cmask
        uad1, uad2, uad3, uad4 = self.encoder(UC[0], UC[1])
        ucf, ucmask = self.decoder(uad1, uad2, uad3, uad4)

        # synthesis the change feature
        # rcm1 = F.upsample_bilinear(cmask, ad1.size()[2:])
        # rcm2 = F.upsample_bilinear(cmask, ad2.size()[2:])
        # rcm3 = F.upsample_bilinear(cmask, ad3.size()[2:])
        # rcm4 = F.upsample_bilinear(cmask, ad4.size()[2:])
        # rcm5 = F.upsample_bilinear(cmask, ad5.size()[2:])
        


        sc_ad1 = ad1 + uad1
        sc_ad2 = ad2 + uad2
        sc_ad3 = ad3 + uad3
        sc_ad4 = ad4 + uad4
        # sc_ad5 = ad5 + uad5



        scf, scmask  = self.decoder(sc_ad1, sc_ad2, sc_ad3, sc_ad4)
        # sucf, sucmask = self.decoder(suc_ad1, suc_ad2, suc_ad3, suc_ad4, suc_ad5)

        return cmask, ucmask, scmask, cf, ucf, scf




class DiscriminatorF(nn.Module):
    def __init__(self,opt=None):
        super(DiscriminatorF, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AvgPool2d((2, 2)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output






class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=13, output_nc=3, num_downs=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.extractor = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=input_nc, submodule=unet_block, norm_layer=norm_layer)
        
        self.model1 = UnetSkipConnectionBlock(output_nc, ngf, input_nc=ngf, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.model2 = UnetSkipConnectionBlock(output_nc, ngf, input_nc=ngf, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.model3 = UnetSkipConnectionBlock(output_nc, ngf, input_nc=ngf, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.model4 = UnetSkipConnectionBlock(output_nc, ngf, input_nc=ngf, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, uimg1, uimg2, img1, img2, mask):
        """Standard forward"""
        feat = self.extractor(torch.cat([uimg1, uimg2, img1, img2, mask], 1))
        print("---------------------------")
        print(feat.shape)
        u1 = self.model1(feat)
        u2 = self.model2(feat)
        c1 = self.model3(feat)
        c2 = self.model4(feat)

        return u1, u2, c1, c2


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)



































