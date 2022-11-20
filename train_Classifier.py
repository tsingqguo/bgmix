import os
import time
import math
import torch
import numpy as np
from torchvision import models
import torch.nn as nn
from options import CDTrainOptions2

from utils import *
from torch.optim import SGD, Adam, lr_scheduler
from models.vgg16 import VGG16_C
device = "cuda" if torch.cuda.is_available() else "cpu"
from datasets import *
from tqdm import tqdm
import argparse



NUM_CHANNELS = 3


def get_loader(args):


    dataset_train = os.path.join(args.datadir, "train")
    dataset_test = os.path.join(args.datadir, "test")

    train_loader = Get_dataloader(dataset_train, batch=args.batch_size, reshape_size=(256, 256), model="SC")
    test_loader = Get_dataloader(dataset_test, batch=args.batch_size,  reshape_size=(256, 256), model="SC")

    return train_loader, test_loader




def train(args, model):
    NUM_CLASSES = args.num_classes  # pascal=21, cityscapes=20
    savedir = args.model_result_dir
    weight = torch.ones(NUM_CLASSES)
    train_loader, test_loader = get_loader(args)

    if args.cuda:
        criterion = CrossEntropyLoss2d(weight).cuda()

    else:
        criterion = CrossEntropyLoss2d(weight)
        # save log
    automated_log_path = savedir + "/automated_log.txt"
    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write(
                "Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")
    paras = dict(model.named_parameters())
    paras_new = []

    for k, v in paras.items():

        if 'bias' in k:
            if 'dec' in k:
                paras_new += [{'params': [v], 'lr': 0.02 *
                               args.lr, 'weight_decay': 0}]
            else:
                paras_new += [{'params': [v], 'lr': 0.2 *
                               args.lr, 'weight_decay': 0}]
        else:
            if 'dec' in k:
                paras_new += [{'params': [v], 'lr': 0.01 *
                               args.lr, 'weight_decay': 0.00004}]
            else:
                paras_new += [{'params': [v], 'lr': 0.1 *
                               args.lr, 'weight_decay': 0.00004}]
    optimizer = Adam(paras_new, args.lr, (0.9, 0.999),
                     eps=1e-08, weight_decay=1e-4)

    def lambda1(epoch): return pow((1 - ((epoch - 1) / args.num_epochs)), 0.9)
    # learning rate changed every epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    start_epoch = 1

    for epoch in range(start_epoch, args.num_epochs + 1):
        tbar = tqdm(train_loader, desc='\r')
        print("----- TRAINING - EPOCH", epoch, "-----")
        scheduler.step(epoch)
        epoch_loss = []
        time_train = []
        usedLr = 0
        # for param_group in optimizer.param_groups:
        for param_group in optimizer.param_groups:
            # print("LEARNING RATE:", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.cuda().train()

        for step, (images, images2,img1_num, img2_num, labels, name) in enumerate(tbar):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                images2 = images2.cuda()
                labels = torch.tensor(labels).cuda()

            img1_aug, img2_aug = aug(img1_num, img2_num)
            inputs = Variable(images)
            inputs2 = Variable(images2)
            img1_aug = Variable(img1_aug)
            img2_aug = Variable(img2_aug)
            targets = Variable(labels)

            feats, out = model(inputs, inputs2)
            feats1, out1 = model(img1_aug, img2_aug)

            loss = criterion(out, targets)
            loss1 = criterion(out1, targets)
            loss += loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                # print('loss: {} (epoch: {}, step: {})'.format(average, epoch, step),
                #       "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
                tbar.set_description('loss: %.8f | epoch: %d | step: %d | Avg time/img: %.4f' % (
                average, epoch, step, sum(time_train) / len(time_train) / args.batch_size))
        localtime = time.asctime(time.localtime(time.time()))
        print(localtime)
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        iouTrain = 0

        # calculate eval-loss and eval-IoU
        #average_epoch_loss_val, iouVal = eval(args, model, loader_val, criterion, epoch)
        average_epoch_loss_val = 0
        iouVal = 0
        # save model every X epoch
        if epoch % args.epoch_save == 0:
            torch.save(model.state_dict(), '{}_{}.pth'.format(
                os.path.join(args.model_result_dir, args.exp_name), str(epoch)))
        t = time.strftime("%D:%H:%m:%S")
        # save log
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%s\t\t%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
                t,epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr))

    return (model)


def main(args):
    '''
        Train the model and record training options.
    '''
    savedir = '{}'.format(args.model_result_dir)
    modeltxtpath = os.path.join(savedir, 'model.txt')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/opts.txt', "w") as myfile:  # record options
        myfile.write(str(args))

    # initialize the network
    model = VGG16_C(num_classes=args.num_classes)


    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True

    with open(modeltxtpath, "w") as myfile:  # record model
        myfile.write(str(model))

    if args.cuda:
        model = model.cuda()
        print("---------cuda--------")
    # checkpoint = torch.load(args.pretrained)
    # model.load_state_dict(checkpoint)
    model.train()

    print("========== TRAINING ===========")
    train(args, model)
    print("========== TRAINING FINISHED ===========")



if __name__ == '__main__':
    parser = CDTrainOptions4().parse()
    main(parser)
