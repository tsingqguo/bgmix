import argparse
import os
import torch

class CDTrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--cuda', action='store_true', default=True)
        self.parser.add_argument('--datadir', type=str, default='/home/wrf/4TDisk/RS/AICD/new_aicd/AICD_AUG3/data_AICD3/CD/',
                                 help='the txt path of the changed images and labels')
        self.parser.add_argument('--data-test-path', type=str, default='/home/wrf/4TDisk/RS/AICD/new_aicd/AICD_AUG3/data_AICD3/CD/test_C/',
                                 help='the txt path of the testing images and labels')
        self.parser.add_argument('--label-path', type=str, default='/home/wrf/4TDisk/RS/AICD/new_aicd/AICD_AUG3/test/all/CSCEN/label/',
                                 help='the path of test label ')
        self.parser.add_argument('--dataset_name', type=str, default="AICD", help='name of the dataset')
        self.parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
        self.parser.add_argument('--img_height', type=int, default=256, help='size of image height')
        self.parser.add_argument('--img_width', type=int, default=256, help='size of image width')
        self.parser.add_argument('--img_result_dir', type=str, default='result_images/time', help=' where to save the result images')
        self.parser.add_argument('--resume', action='store_true', help="load the pretrained pth from model_result_dir")
        self.parser.add_argument('--model_result_dir', type=str, default='saved_models/time', help=' where to save the checkpoints')
        self.parser.add_argument('--resume-SC', type=str, default='./saved_models/AICD3_SC/SCGD_100.pth',
                                 help='the path of saved classifier checkpoints')

        #-------------------------------------------------------------------------------------------------#
        # additional params for classifier
        self.parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
        self.parser.add_argument('--epoch-save', type=int, default=10)  # You can use this value to save model every X epoch
        self.parser.add_argument('--steps-loss', type=int, default=100)
        self.parser.add_argument('--num-epochs', type=int, default=100) # just for classifier training
        self.parser.add_argument('--num-classes', type=int, default=2)

    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()

        os.makedirs('%s' % ( args.img_result_dir), exist_ok=True)
        os.makedirs('%s' % ( args.model_result_dir), exist_ok=True)

        print('------------ Options -------------')
        with open("%s/args.log" % (args.model_result_dir) ,"w") as args_log:
            for k, v in sorted(vars(args).items()):
                print('%s: %s ' % (str(k), str(v)))
                args_log.write('%s: %s \n' % (str(k), str(v)))

        print('-------------- End ----------------')

        self.args = args
        return self.args







