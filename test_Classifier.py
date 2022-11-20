import os
import time
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
from datasets import *
from options import CDTrainOptions2
from utils import *
from models.vgg16 import VGG16_C
import argparse
import torchvision.transforms as transforms



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset_test = os.path.join(args.datadir, "test")
    loader = Get_dataloader(dataset_test, batch=1, reshape_size=(256, 256), model="SC")

    model = VGG16_C(num_classes=args.num_classes)
    if args.cuda:
        model = model.cuda()

    checkpoint = torch.load(args.resume_SC)
    model.load_state_dict(checkpoint)
    model.eval()
    count = 0
    countp0 = 0
    countp1 = 0
    for image, image2, img1_numpy, img2_numpy, label, name in loader:

        if args.cuda:
            images = image.cuda()
            images2 = image2.cuda()
            label = label.cuda()
        inputs = Variable(images, volatile=True)
        inputs2 = Variable(images2, volatile=True)
        label = Variable(label, volatile=True)

        feats, out = model(inputs, inputs2)
        _, pred = torch.max(out, dim=1)

        count += 1

        if label.item() == int(pred):
            countp0 += 1
        else:
            countp1 += 1
        print("This is the {}th of image!".format(count))
    # f1.close()

    print("ACC:{}".format(countp0/count))

if __name__ == '__main__':
    parser = CDTrainOptions2().parse()
    main(parser)


