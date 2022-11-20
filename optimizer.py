import torch
# Optimizers
def Get_optimizers(args, generator, discriminator,discriminator2):

    optimizer_G = torch.optim.SGD(
        generator.parameters(),
        lr=args.lr, momentum=0.5)

    optimizer_D = torch.optim.SGD(
        discriminator.parameters(),
        lr=args.lr, momentum=0.5)
    optimizer_D2 = torch.optim.SGD(
        discriminator2.parameters(),
        lr=args.lr,  momentum=0.5)


    return optimizer_G, optimizer_D, optimizer_D2





def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
