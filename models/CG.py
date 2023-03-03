import torch
import torch.nn as nn
# import torch.nn.init as init
import torch.nn.functional as F
# from torch.utils import model_zoo
from torchvision import models
# from .vit import ViT
# from .swin import SwinTransformer
# import functools



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





class Discriminator2(nn.Module):
    def __init__(self,opt=None):
        super(Discriminator2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
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

    def forward(self, x, y):
        output = self.net(torch.cat([x, y], 1))
        return output
































