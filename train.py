from loss import *
from models.models import Create_nets
from models.vgg16 import VGG16_C
from datasets import *
from options import CDTrainOptions
from optimizer import *
from test import test
from eval import eval1
from utils import *
from tqdm import tqdm
import time
from torch.autograd import Variable
import augmentations
from manual_seed import set_seed, adjust_lr
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


def BGAware(img1, img2, uimg1, uimg2, mask):
    weight = np.float32(np.random.dirichlet([1]*3))
    # weight2 = np.float32(np.random.dirichlet([1]))
    m = np.float32(np.random.beta(1, 1))
    # n = np.float32(np.random.beta(1, 1))

    # aug1 = []
    # aug2 = []
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    # B,_,_,_ = img1.shape
    # for i in range(4):  # output dimension
    mix1 = img1
    mix2 = img2
    # print(img1)
    mixed1 = np.zeros_like(img1)
    mixed2 = np.zeros_like(img1)
    # mix2 = np.zeros_like(img2[0])
    width = 3
    for j in range(width):
        depth = np.random.randint(1, 4)
        for _ in range(depth):  # mixed depth
            op = np.random.randint(0, 4)
            im1 = np.array(uimg1[op]).astype(np.float32)
            im2 = np.array(uimg2[op]).astype(np.float32)
            # mix1 = mask * mix1 + im1 * (1-mask)
            # mix2 = mask * mix2 + im2 * (1-mask)
            mix1 = mask * mix1 * (1-m) + im1 * (1-mask) * m
            mix2 = mask * mix2 * (1-m) + im2 * (1-mask) * m
        mixed1 += weight[j] * mix1
        mixed2 += weight[j] * mix2
    # aug1.append(mixed1)
    # aug2.append(mixed1)
    # print(mixed1.shape)
    return Image.fromarray(np.uint8(mixed1)), Image.fromarray(np.uint8(mixed2))
    # return torch.stack(aug1), torch.stack(aug2)



def BGMix(image1,image2, uimage1, uimage2, mask):

    preprocess = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    aug_list = augmentations.augmentations

    ws = np.float32(
        np.random.dirichlet([1] * 3))
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []

    B,_,_,_ = images.shape
    for index in range(B):
        aug = np.random.choice([True, False])
        img1 = Image.fromarray(image1[index].numpy())
        img2 = Image.fromarray(image2[index].numpy())
        cdm = mask[index].detach().cpu().numpy().transpose(1, 2, 0)

        img1_tensor = preprocess(img1)
        img2_tensor = preprocess(img2)

        if aug:

            mix = torch.zeros_like(img1_tensor)
            mix2 = torch.zeros_like(img2_tensor)

            for i in range(3): # three paths
                image_aug = img1.copy()
                image_aug2 = img2.copy()

                depth = np.random.randint(1, 4)
                for _ in range(depth):
                    idx = np.random.choice([0, 1])
                    if idx == 0:
                        op = np.random.choice(aug_list)
                        image_aug = op(image_aug, 1)
                        image_aug2 = op(image_aug2, 1)
                
                    else:
                        # augamented by UIP
                        image_aug, image_aug2 = BGAware(image_aug, image_aug2, uimage1, uimage2, cdm)

                # Preprocessing commutes since all coefficients are convex
            
                mix += ws[i] * preprocess(image_aug)
                mix2 += ws[i] * preprocess(image_aug2)
            
            
            mixed = (1 - m) * img1_tensor + m * mix
            mixed2 = (1 - m) * img2_tensor + m * mix2

            aug1.append(mixed)
            aug2.append(mixed2)

        else:
            aug1.append(img1_tensor)
            aug2.append(img2_tensor)



    cg1, cg2 = torch.stack(aug1).cuda(), torch.stack(aug2).cuda()  # change imag

    return cg1, cg2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(1)
    #load the args
    args = CDTrainOptions().parse()

    C_train = os.path.join(args.datadir, "train_C")
    cloder1 = Get_dataloader(C_train, args.batch_size, reshape_size=(args.img_height, args.img_width), model="CD")
    UC_train = os.path.join(args.datadir, "train_UC")
    dis_UC_loder = Get_dataloader(UC_train, args.batch_size, reshape_size=(args.img_height, args.img_width), model="CD")

    real = iter(cloder1)
    dis_UC = iter(dis_UC_loder)

    
    cls_model = VGG16_C()
    for p in cls_model.parameters():
        p.requires_grad = False

  
    cls_model.to(device)
 
    cls_model.load_state_dict(torch.load(args.resume_SC))
    cls_model.eval()



    j=0
    # 开始训练
    # Initialize generator and discriminator
    generator, discriminator, discriminator2 = Create_nets(args)
    # Loss functions
    criterion_GAN, criterion_pixelwise = Get_loss_func(args) 
    mssim_l1 = MS_SSIM_L1_LOSS()
    # Optimizers
    optimizer_G, optimizer_D, optimizer_D2 = Get_optimizers(args, generator, discriminator, discriminator2)
    log = {'bestmae_iter': 0, 'best_mae': 10, 'fm': 0, 'bestfm_it': 0, 'best_fm': 0, 'mae': 0, 'R':0, "P":0}
    log_file = open('%s/train_log.txt' % (args.model_result_dir), 'w')
    f = open('%s/best.txt' % (args.model_result_dir), 'w')
    tbar = tqdm(range(args.epoch_start, 100000))
    for i in tbar:
        try:
            # --------img1 img2 label name ---------
            img1, img2,img1_numpy,img2_numpy, _, name = next(real)
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # real_image = real_image.to(device)

            dis_UC_img1, dis_UC_img2, dis_UC_img1_numpy, dis_UC_img2_numpy,_, uname = next(dis_UC)
            dis_UC_img1 = dis_UC_img1.to(device)
            dis_UC_img2 = dis_UC_img2.to(device)

        except (OSError, StopIteration):

            # --------img1 img2 label name ---------
            real = iter(cloder1)
            dis_UC = iter(dis_UC_loder)
            img1, img2, img1_numpy,img2_numpy,_, name = next(real)
            img1 = img1.to(device)
            img2 = img2.to(device)

            dis_UC_img1, dis_UC_img2, dis_UC_img1_numpy, dis_UC_img2_numpy,_, uname = next(dis_UC)
            dis_UC_img1 = dis_UC_img1.to(device)
            dis_UC_img2 = dis_UC_img2.to(device)
        #plr = args.lr
        plr = adjust_lr(optimizer_G, i, args.lr)
        adjust_lr(optimizer_D, i, args.lr)
        adjust_lr(optimizer_D2, i, args.lr)
        # ------------------
        #  Train Generators
        # ------------------
        # Adversarial ground truths
        patch=(1,1,1)
        ones = Variable(torch.FloatTensor(np.ones((img1.size(0), *patch))).cuda(), requires_grad=False)
        zeros = Variable(torch.FloatTensor(np.zeros((img1.size(0), *patch))).cuda(), requires_grad=False)
      
        optimizer_G.zero_grad()
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(discriminator2, False)
        
        oc, ouc, sc, ocf, oucf, scf = generator([img1, img2], [dis_UC_img1, dis_UC_img2])
        Mask1 = oc
        zerol = Variable(torch.zeros_like(ouc).cuda(), requires_grad=False)
        img1_aug, img2_aug = BGMix(img1_numpy, img2_numpy, dis_UC_img1_numpy, dis_UC_img2_numpy, Mask1)

        oc_g, ouc_g, sc_g, ocf_g, oucf_g, scf_g = generator([img1_aug, img2_aug], [dis_UC_img1, dis_UC_img2])
        Mask2 = oc_g
            
        syn_img_C1 = Mask1 * img1 + (1 - Mask1) * dis_UC_img1
        syn_img_C2 = Mask1 * img2 + (1 - Mask1) * dis_UC_img2

        syn_img_UC1 = Mask1 * dis_UC_img1 + (1 - Mask1) * img1
        syn_img_UC2 = Mask1 * dis_UC_img2 + (1 - Mask1) * img2

        syn_img_C1_g = Mask2 * img1_aug + (1 - Mask2) * dis_UC_img1
        syn_img_C2_g = Mask2 * img2_aug + (1 - Mask2) * dis_UC_img2

        syn_img_UC1_g = Mask2 * dis_UC_img1 + (1 - Mask2) * img1_aug
        syn_img_UC2_g = Mask2 * dis_UC_img2 + (1 - Mask2) * img2_aug

        mssi_l1loss1 = 1 - mssim_l1(img1, syn_img_C1)
        mssi_l1loss2 = 1 - mssim_l1(dis_UC_img2, syn_img_UC2)

        mssi_l1loss1g = 1 - mssim_l1(img1_aug, syn_img_C1_g)
        mssi_l1loss2g = 1 - mssim_l1(dis_UC_img2, syn_img_UC2_g)

        pred_fake = discriminator(syn_img_C1, syn_img_C2)
        loss_GAN1 = criterion_GAN(pred_fake, zeros)
        pred_fake2 = discriminator2(syn_img_UC1, syn_img_UC2)
        loss_GAN2 = criterion_GAN(pred_fake2, zeros)

        pred_fakeg = discriminator(syn_img_C1_g, syn_img_C2_g)
        loss_GAN1g = criterion_GAN(pred_fakeg, zeros)
        pred_fake2g = discriminator2(syn_img_UC1_g, syn_img_UC2_g)
        loss_GAN2g = criterion_GAN(pred_fake2g, zeros)



        loss_umask1 = criterion_pixelwise(ouc, zerol)
        loss_umask1g = criterion_pixelwise(ouc_g, zerol)
        loss_cmse = criterion_pixelwise(sc, oc)
        loss_cmseg = criterion_pixelwise(sc_g, oc_g)

    
        loss_c = cosins(cls_model, [syn_img_C1, syn_img_C2], [img1, img2])
        loss_uc = cosins(cls_model, [syn_img_UC1, syn_img_UC2], [dis_UC_img1, dis_UC_img2])
        loss_cg = cosins(cls_model, [syn_img_C1_g, syn_img_C2_g], [img1_aug, img2_aug])
        loss_ucg = cosins(cls_model, [syn_img_UC1_g, syn_img_UC2_g], [dis_UC_img1, dis_UC_img2])
        loss_auc = cosins(cls_model, [img1_aug, img2_aug], [img1, img2])
        loss_cosin = loss_c + loss_uc + loss_cg + loss_ucg + loss_auc
        loss_G = loss_GAN1+loss_GAN2 + 0.01*loss_umask1 + loss_cosin + 5 * (mssi_l1loss1 + mssi_l1loss2) + loss_cmse +loss_GAN1g+loss_GAN2g + 0.01*loss_umask1g + 5*(mssi_l1loss1g + mssi_l1loss2g)+loss_cmseg

        loss_G.backward()
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # # ---------------------
        optimizer_D.zero_grad()
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(discriminator2, False)
        # Real loss
        pred_real = discriminator(img1, img2)
        loss_real = criterion_GAN(pred_real, zeros)

        pred_realg = discriminator(img1_aug, img2_aug)
        loss_realg = criterion_GAN(pred_realg, zeros)

        # Fake loss
        pred_fake = discriminator(syn_img_C1.detach(), syn_img_C2.detach())
        loss_fake = criterion_GAN(pred_fake, ones)
        pred_fakeg = discriminator(syn_img_C1_g.detach(), syn_img_C2_g.detach())
        loss_fakeg = criterion_GAN(pred_fakeg, ones)

        loss_D1 = (loss_real + loss_fake+loss_realg + loss_fakeg)
        loss_D1.backward()
        optimizer_D.step()

        optimizer_D2.zero_grad()
        requires_grad(generator, False)
        requires_grad(discriminator, False)
        requires_grad(discriminator2, True)
        # Real loss
        pred_real2 = discriminator2(dis_UC_img1, dis_UC_img2)
        loss_real2 = criterion_GAN(pred_real2, zeros)

        # Fake loss
        pred_fake2 = discriminator2(syn_img_UC1.detach(), syn_img_UC2.detach())
        loss_fake2 = criterion_GAN(pred_fake2, ones)
        pred_fake2g = discriminator2(syn_img_UC1_g.detach(), syn_img_UC2_g.detach())
        loss_fake2g = criterion_GAN(pred_fake2g, ones)

        
        loss_D2 = (loss_real2 + 0.5*(loss_fake2 + loss_fake2g))
        # loss_D2 = (loss_real2 + loss_fake2 + loss_fake2g)
        loss_D2.backward()
        optimizer_D2.step()


        localtime = time.asctime(time.localtime(time.time()))
        tbar.set_description('Time: %s | Iter: %d | Dl1: %f | Dl2: %f | loss_G: %f loss_GAN1:%f | loss_GAN2:%f'
                                % (localtime, i, loss_D1.data.cpu(),loss_D2.data.cpu(),
                                loss_G.data.cpu(), loss_GAN1.data.cpu(),loss_GAN2.data.cpu()))
        log_file.write('Time: %s | Iter: %d | Dl1: %f | Dl2: %f | loss_G: %f loss_GAN1:%f | loss_GAN2:%f\n'
                        % (localtime, i, loss_D1.data.cpu(),loss_D2.data.cpu(),loss_G.data.cpu(), loss_GAN1.data.cpu(),loss_GAN2.data.cpu()))
        log_file.flush()
    
        
        if i % 1000==0:
            image_path = '%s/%s' % (args.img_result_dir, str(i))
            os.makedirs(image_path, exist_ok=True)
            to_image(img1, path=image_path, name="C1_"+str(name[0]))
            to_image(img2, path=image_path, name="C2_"+str(name[0]))
            to_image(dis_UC_img1, path=image_path, name="U1_"+str(uname[0]))
            to_image(dis_UC_img2, path=image_path, name="U2_"+str(uname[0]))

            to_image(syn_img_C1, name="syn_c1_"+str(name[0]), path=image_path)
            to_image(syn_img_C2, name="syn_c2_"+str(name[0]), path=image_path)
            to_image(syn_img_UC1, name="syn_uc1_"+str(name[0]), path=image_path)
            to_image(syn_img_UC2, name="syn_uc2_"+str(name[0]), path=image_path)
            to_image_mask(oc, name="mask_"+str(name[0]), path=image_path)


        if args.checkpoint_interval != -1 and i != 0 and i % 1000 == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), '%s/generator_latest.pth' % (args.model_result_dir))
            torch.save(discriminator.state_dict(), '%s/discriminator1_latest.pth' % (args.model_result_dir))
            torch.save(discriminator2.state_dict(), '%s/discriminator2_latest.pth' % (args.model_result_dir))

            mask_save_path = '%s/test/' % (args.model_result_dir)
            image_path = args.data_test_path
            test(generator, mask_save_path, image_path)
            gt_path = args.label_path
            mae1, fmeasure1, recall, precision = eval1(mask_save_path, gt_path, 2)
            
            if fmeasure1 > log['best_fm']:
                log['bestfm_iter'] = i
                log['best_fm'] = fmeasure1
                log['mae'] = mae1
                log['R'] = recall
                log['P'] = precision
                torch.save(generator.state_dict(), '%s/generator_best.pth' % (args.model_result_dir))
                torch.save(discriminator.state_dict(), '%s/discriminator1_best.pth' % (args.model_result_dir))
                torch.save(discriminator2.state_dict(), '%s/discriminator2_best.pth' % (args.model_result_dir))
            # print('====================================================================================================================')
            # print('Iter:', i, "mae:", mae1, "fmeasure:", fmeasure1, "R:",recall, "P:",precision)
            # print('bestfm_iter:', log['bestfm_iter'], 'mae:', log['mae'], 'best_fm:', log['best_fm'], "R:", log['R'], "P:", log['P'])
            # print('====================================================================================================================')

            f.write('====================================================================================================================\n')
            f.write("Iter: {}, mae: {}, fmeasure: {}, R: {}, P: {}\n".format(i, mae1, fmeasure1, recall, precision))
            f.write("bestfm_iter: {}, mae: {}, best_fm: {}, R: {}, P: {}\n".format(log['bestfm_iter'], log['mae'], log['best_fm'], log['R'], log['P']))
            f.write('====================================================================================================================\n\n')
            f.flush()
    f.close()


if __name__ == "__main__":
    main()
