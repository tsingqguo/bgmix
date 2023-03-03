import torch.nn as nn
import torch
from models.DG import GeneratorD, Discriminator2, Discriminator2C
from models.TG import GeneratorT
from models.CG import GeneratorVGGC, DiscriminatorCU, WSCD, WSCD2, WSCD4,WSCD5, DiscriminatorF, WSCD2_sub, WSCD2_ViT, WSCD2_Swin
####################################################
# Initialize generator and discriminator
####################################################
def Create_nets(args):
    # generator = GeneratorD()
    generator = WSCD2()
    # generator = GeneratorT()
    discriminator = Discriminator2()
    discriminator2=Discriminator2()
    # discriminator3=Discriminator2()

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        discriminator2= discriminator2.cuda()
        # discriminator3=discriminator3.cuda()
    if args.epoch_start != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load(args.resume), False)
        # generator.load_state_dict(torch.load('%sgenerator_best.pth' % (args.resume)), False)
        # discriminator.load_state_dict(torch.load('%sdiscriminator1_best.pth' % (args.resume)))
        # discriminator2.load_state_dict(torch.load('%sdiscriminator2_best.pth' % (args.resume)))
        # generator.load_state_dict(torch.load('log/%s-%s/%s/generator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,args.epoch_start)))
        # discriminator.load_state_dict(torch.load('log/%s-%s/%s/discriminator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,args.epoch_start)))
        # discriminator2.load_state_dict(torch.load('log/%s-%s/%s/discriminator2_%d.pth' % (
        # args.exp_name, args.dataset_name, args.model_result_dir, args.epoch_start)))

    return generator, discriminator,discriminator2
    # return generator, discriminator,discriminator2,discriminator3

if __name__ == '__main__':
    # c = Discriminator()
    print()