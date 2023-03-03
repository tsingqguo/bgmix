import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from PIL import Image

def sample_images(generator, test_dataloader, args, epoch, batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs['A'].type(torch.FloatTensor).cuda())
    real_B = Variable(imgs['B'].type(torch.FloatTensor).cuda())
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, '%s-%s/%s/%s-%s.png' % (args.exp_name, args.dataset_name, args.img_result_dir, batches_done, epoch), nrow=5, normalize=True)

def to_image(tensor,name,path):
    #for i in range(32):
    image = tensor
    if not os.path.isdir(path):
        os.makedirs(path)
    fake_samples_file = path+'/' + name
    save_image(image.detach(),
               fake_samples_file,
               normalize=True,
               range=(-1.,1.),
               nrow=4)

#保留mask
def to_image_mask(tensor, name, path):
    image = tensor  # [i].cpu().clone()
    if not os.path.isdir(path):
        os.makedirs(path)
    fake_samples_file = path + '/' + name
    save_image(image.detach(),
               fake_samples_file,
               normalize=True,
               range=(0., 1.),
               nrow=4)


def to_image_test(tensor, path, name):
    mask = tensor.detach().cpu().numpy()[0,0,:,:]  # [i].cpu().clone()
    # print(mask.shape)
    if not os.path.isdir(path):
        os.makedirs(path)
    # fake_samples_file = path + '/{}.bmp'.format(str(i))
    fake_samples_file = os.path.join(path, name)
    mask = Image.fromarray(mask*255).convert('L')
    mask.save(fake_samples_file)
    # save_image(image.detach(),
    #            fake_samples_file,
    #            normalize=True,
    #            range=(0., 1.),
    #            nrow=4)

