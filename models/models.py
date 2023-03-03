import torch.nn as nn
import torch

from models.CG import WSCD2, Discriminator2

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
    if args.resume:
        # Load pretrained models
        generator.load_state_dict(torch.load('%s/generator_best.pth' % (args.model_result_dir)))
        discriminator.load_state_dict(torch.load('%s/discriminator1_best.pth' % (args.model_result_dir)))
        discriminator2.load_state_dict(torch.load('%s/discriminator2_best.pth' % (args.model_result_dir)))


    return generator, discriminator,discriminator2
    # return generator, discriminator,discriminator2,discriminator3

if __name__ == '__main__':
    # c = Discriminator()
    print()