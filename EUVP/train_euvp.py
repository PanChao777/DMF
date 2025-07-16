
import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import Config.euvp_dataset as dataset
from Config.vgg import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from Config.euvp_options import opt, device
from Config.models import *
from Config.misc import *
import re
import sys
from torchsummary import summary
from numba import cuda
from pytorch_msssim import ssim
import matplotlib.pyplot as plt 
 # 添加matplotlib用于绘图
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置使用Agg后端

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
  
if __name__ == '__main__':

    batches = int(opt.num_images / opt.batch_size)
    
    # 初始化用于记录损失的列表
    mae_losses = []
    ssim_losses = []
    vgg_losses = []
    total_losses = []
    epochs_list = []
    
    dicam = DICAM()
    dicam.to(device)
    summary(dicam,input_size=(3,256,256))

    mae_loss = nn.L1Loss()

    vgg = Vgg19(requires_grad=False).to(device)

    optim_g = optim.Adam(dicam.parameters(), 
                         lr=opt.learning_rate_g, 
                         betas = (opt.beta1, opt.beta2), 
                         weight_decay=opt.wd_g)

    dataset = dataset.Dataset_Load(data_path = opt.data_path,
                                   transform=dataset.ToTensor()
                                   )
    batches = int(dataset.len / opt.batch_size)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)
    
    models_loaded = getLatestCheckpointName()    
    latest_checkpoint_G = models_loaded
    
    print('loading model for DMF ', latest_checkpoint_G)
    
    if latest_checkpoint_G == None :
        start_epoch = 1
        print('No checkpoints found for DMF retraining')
    
    else:
        checkpoint_g = torch.load(os.path.join(opt.checkpoints_dir, latest_checkpoint_G))    
        start_epoch = checkpoint_g['epoch'] + 1
        dicam.load_state_dict(checkpoint_g['model_state_dict'])
        optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
            
        print('Restoring model from checkpoint ' + str(start_epoch))
    
    dicam.train()

    for epoch in range(start_epoch, opt.end_epoch + 1):

        opt.total_mae_loss = 0.0
        opt.total_vgg_loss = 0.0
        opt.total_loss = 0.0
        opt.total_ssim_loss = 0.0
    
        for i_batch, sample_batched in enumerate(dataloader):

            hazy_batch = sample_batched['hazy']
            clean_batch = sample_batched['clean']

            hazy_batch = hazy_batch.to(device)
            clean_batch = clean_batch.to(device)

            
            pred_batch = dicam(hazy_batch)

            batch_mae_loss = torch.mul(opt.lambda_mae, mae_loss(pred_batch, clean_batch))
            batch_mae_loss.backward(retain_graph=True)
            
            batch_ssim_loss = 1-ssim(pred_batch, clean_batch,data_range=1, size_average=True)
            batch_ssim_loss.backward(retain_graph=True)

            clean_vgg_feats = vgg(normalize_batch(clean_batch))
            pred_vgg_feats = vgg(normalize_batch(pred_batch))
            batch_vgg_loss = torch.mul(opt.lambda_vgg, mae_loss(pred_vgg_feats.relu4_3, clean_vgg_feats.relu4_3))
            batch_vgg_loss.backward()
            
            opt.batch_mae_loss = batch_mae_loss.detach().cpu().item()
            opt.total_mae_loss += opt.batch_mae_loss

            opt.batch_ssim_loss = batch_ssim_loss.detach().cpu().item()
            opt.total_ssim_loss += opt.batch_ssim_loss

            opt.batch_vgg_loss = batch_vgg_loss.detach().cpu().item()
            opt.total_vgg_loss += opt.batch_vgg_loss
            
            opt.batch_loss = opt.batch_mae_loss + opt.batch_ssim_loss + opt.batch_vgg_loss
            opt.total_loss += opt.batch_loss
            
            optim_g.step()
            optim_g.zero_grad() 

            print('\r Epoch : ' + str(epoch) + ' | (' + str(i_batch+1) + '/' + str(batches) + ') | l_mae: ' + str(opt.batch_mae_loss/2) + ' | l_ssim: ' + str(1-opt.batch_ssim_loss)+ ' | l_vgg: ' + str(opt.batch_vgg_loss), end='', flush=True)

        # 计算平均损失并记录
        avg_mae = opt.total_mae_loss / batches
        avg_ssim = opt.total_ssim_loss / batches
        avg_vgg = opt.total_vgg_loss / batches
        avg_total = opt.total_loss / batches
        
        mae_losses.append(avg_mae)
        ssim_losses.append(avg_ssim)
        vgg_losses.append(avg_vgg)
        total_losses.append(avg_total)
        epochs_list.append(epoch)
        
        print('\n\nFinished ep. %d, lr = %.6f, mean_mae = %.6f, mean_ssim = %.6f, mean_vgg = %.6f' % (epoch, get_lr(optim_g), (opt.total_mae_loss / batches)/2,1- (opt.total_ssim_loss / batches), opt.total_vgg_loss / batches))

        torch.save({'epoch':epoch, 
                    'model_state_dict':dicam.state_dict(), 
                    'optimizer_state_dict':optim_g.state_dict(), 
                    'mae_loss':opt.total_mae_loss,
                    'ssim_loss':opt.total_ssim_loss, 
                    'vgg_loss':opt.total_vgg_loss, 
                    'opt':opt,
                    'total_loss':opt.total_loss}, os.path.join(opt.checkpoints_dir, 'DICAM_' + str(epoch) + '.pt'))
        
        # 绘制损失曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, mae_losses, label='MAE Loss',linewidth=2)
        plt.plot(epochs_list, ssim_losses, label='SSIM Loss',linewidth=2)
        plt.plot(epochs_list, vgg_losses, label='VGG Loss',linewidth=2)
        plt.plot(epochs_list, total_losses, label='Total Loss', linestyle='--',linewidth=2)
        
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        
        # 保存图像到checkpoints目录
        loss_plot_path = os.path.join(opt.checkpoints_dir, 'loss_curves.png')
        plt.savefig(loss_plot_path)
        plt.close()
        
        print(f"Loss curves saved to {loss_plot_path}")