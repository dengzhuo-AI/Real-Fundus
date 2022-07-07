import os
from config import Config 
opt = Config('train.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
import utils
from RF_dataset import get_training_data_RF, get_validation_data_RF
from pdb import set_trace as stx 

from architecture.RFormer import RFormer_G
from architecture.RFormer import RFormer_D


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


from loss import CharbonnierLoss, PerceptualLoss, EdgeLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR
save_images = opt.TRAINING.SAVE_IMAGES

######### Create Logger ###########
logger = utils.setup_log(
            name='train', log_dir=opt.TRAINING.SAVE_DIR, file_name=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())+'_log.txt')

######### Model ###########

model_g = RFormer_G()
model_d = RFormer_D()
model_g.cuda()
model_d.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
  logger.info('GPUs:{}'.format(gpus))


new_lr_G = 1e-4
new_lr_D = 1e-4
optimizer_G = optim.Adam(model_g.parameters(), lr=new_lr_G, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)
optimizer_D = optim.Adam(model_d.parameters(), lr=new_lr_D, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)


######### Scheduler ###########
warmup = True
if warmup:
    warmup_epochs = 5
    scheduler_cosine_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)
    scheduler_G = GradualWarmupScheduler(optimizer_G, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine_G)
    scheduler_G.step()
    scheduler_cosine_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, opt.OPTIM.NUM_EPOCHS - warmup_epochs,eta_min=1e-6)
    scheduler_D = GradualWarmupScheduler(optimizer_D, multiplier=1, total_epoch=warmup_epochs,after_scheduler=scheduler_cosine_D)
    scheduler_D.step()

if len(device_ids)>1:
    model_g = nn.DataParallel(model_g, device_ids = device_ids)
    model_d = nn.DataParallel(model_d, device_ids=device_ids)

######### Loss ###########
lossc = CharbonnierLoss().cuda()
lossp = PerceptualLoss().cuda()
lossm = nn.MSELoss()
losse = EdgeLoss().cuda()

######### DataLoaders ###########
img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}
train_dataset = get_training_data_RF(train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)
val_dataset = get_validation_data_RF(val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

logger.info('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
logger.info('===> Loading datasets')

mixup = utils.MixUp_AUG()
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)//4 - 1
logger.info(f"Evaluation after every {eval_now} Iterations !!!")

real_labels_patch = Variable(torch.ones(opt.OPTIM.BATCH_SIZE, 169) - 0.05).cuda()
fake_labels_patch = Variable(torch.zeros(opt.OPTIM.BATCH_SIZE, 169)).cuda()

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    D_loss_sum_patch = 0
    train_id = 1
        
    for i, data in enumerate(tqdm(train_loader), 0):

        target = data[0].cuda()
        input_ = data[1].cuda()

        if epoch>5:
            target, input_ = mixup.aug(target, input_)

        restored = model_g(input_)
        restored = torch.clamp(restored,0,1)
        real_output = model_d(target)
        if len(real_labels_patch) > len(real_output):
            D_loss_real = lossm(real_output, real_labels_patch[:len(real_output)])
        else:
            D_loss_real = lossm(real_output, real_labels_patch)
        
        fake_output = model_d(restored)
        if len(fake_labels_patch) > len(fake_output):
            D_loss_fake = lossm(fake_output, fake_labels_patch[:len(fake_output)])
        else:
            D_loss_fake = lossm(fake_output, fake_labels_patch)
        D_loss_patch = D_loss_real + D_loss_fake
        D_loss_sum_patch += D_loss_patch.item()
        optimizer_D.zero_grad()  
        D_loss_patch.backward()
        optimizer_D.step()

        ########################################################################
        restored = model_g(input_)
        restored = torch.clamp(restored, 0, 1)
        fake_output = model_d(restored)
        real_output = model_d(target)
        if len(fake_labels_patch) > len(fake_output):
            G_loss_patch = lossm(fake_output, real_labels_patch[:len(real_output)])
        else:
            G_loss_patch = lossm(fake_output, real_labels_patch)
        loss1 = lossc(restored, target)
        loss2 = lossp(restored, target)
        loss3 = losse(restored, target)
        loss = 1*loss1 + 0.06*loss2 + 0.05*loss3 + 0.2*G_loss_patch
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()
        epoch_loss += loss.item()

        #### Evaluation ####
        if i%eval_now==0 and i>0:
            if save_images:
                utils.mkdir(result_dir + '%d/%d'%(epoch,i))
            model_g.eval()
            with torch.no_grad():
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]

                    restored = model_g(input_)
                    restored = torch.clamp(restored,0,1) 
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))

                    if save_images:
                        target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
                        input_ = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
                        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                        
                        for batch in range(input_.shape[0]):
                            temp = np.concatenate((input_[batch]*255, restored[batch]*255, target[batch]*255),axis=1)
                            utils.save_img(os.path.join(result_dir, str(epoch), str(i), filenames[batch][:-4] +'.jpg'),temp.astype(np.uint8))

                psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
                
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_g.state_dict(),
                                'optimizer' : optimizer_G.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                logger.info("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
            
            model_g.train()
    scheduler_D.step()
    scheduler_G.step()
    
    logger.info("------------------------------------------------------------------")
    logger.info("Epoch: {}\tTime: {:.4f}\tLoss_EPOCH: {:.4f}\tLoss_G: {:.4f}\tLoss_L1: {:.4f}\tLoss_p: {:.6f}\tLoss_E: {:.4f}\tLoss_Gpatch: {:.4f}\tLoss_D: {:.4f}\tLearningRate_D {:.6f}\tLearningRate_G {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss,loss,loss1,loss2,loss3,G_loss_patch, D_loss_patch,scheduler_D.get_lr()[0],scheduler_G.get_lr()[0]))
    logger.info("------------------------------------------------------------------")


    torch.save({'epoch': epoch, 
                'state_dict': model_g.state_dict(),
                'optimizer' : optimizer_G.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    torch.save({'epoch': epoch, 
                'state_dict': model_g.state_dict(),
                'optimizer' : optimizer_G.state_dict()
                }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 


