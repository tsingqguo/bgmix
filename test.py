import argparse
import os
import torch
from tqdm import tqdm
from utils import to_image_test
from torch.autograd import Variable
from datasets import Get_dataloader_test
from models.CG import WSCD2

device = torch.device("cuda:0")

def test(model,mask_save_path,image_path):
    model.eval()
    dataloder = Get_dataloader_test(image_path, 1, reshape_size=(256,256), model="SGCD")
    with torch.no_grad():
        for i, (img1, img2, label, name) in tqdm(enumerate(dataloder)):
            if not torch.cuda.is_available():
                img1 = Variable(img1)
                img2 = Variable(img2)
            else:
                img1 = Variable(img1).cuda()
                img2 = Variable(img2).cuda()
            mask = model(C=[img1, img2], UC=None, test=True)
            # mask = model(img1, img2, test=True)
            os.makedirs(mask_save_path, exist_ok=True)
            to_image_test(mask, path=mask_save_path, name=str(name[0].replace(".jpg", ".png")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--stict', default='log/SCGD-AICD/saved_models/train_WSCD_iter_v10_Aug5_v10_uc03/generator_best.pth', type=str)
    parser.add_argument('--image_path', default='./data_AICD3/CD/test_C', type=str)
    parser.add_argument('--mask_save_path', default='./results/train_WSCD_iter_v10_Aug5_v10_uc03', type=str)
    args=parser.parse_args()

    generator2 = WSCD2().cuda()
    generator2.load_state_dict(torch.load(args.stict))
    generator2.eval()
    test(generator2,args.mask_save_path,args.image_path)




























