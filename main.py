import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datetime import datetime

import glob
import argparse

import numpy as np

import torch
import torch.multiprocessing as mp
import bitsandbytes as bnb

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.dataloader import DataLoader
from model_datasets import CreateDatasets
from split_data import split_data

#import torch.optim as optim
import torch.nn as nn
from lpips import LPIPS
from pytorch_msssim import SSIM

from models import GeneratorUNet, Discriminator
from models import weights_init_normal
from trainer import inverse_transform
# Import the modified train_one_epoch function
from trainer import train_one_epoch_with_accumulation as train_one_epoch
from trainer import val_one_epoch

def main(opt):
    data_path = opt.dataPath
    weight_path = opt.savePath
    print_every = opt.every
    epochs = opt.epoch
    img_size = opt.imgsize
    train_batch_size = opt.batch
    val_batch_size = opt.batch
    use_amp = opt.amp
    accumulation_steps = opt.grad_accu  # Get the accumulation steps from the command line

    if not os.path.exists(opt.savePath):
        os.mkdir(opt.savePath)

    # 若没有cuda设备，则使用cpu运算，且强制禁止AMP加速
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        use_amp = False
    else:
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # 实例化网络
    pix_G = GeneratorUNet().to(device)
    pix_D = Discriminator().to(device)
    pix_G.apply(weights_init_normal)
    pix_D.apply(weights_init_normal)

    start_epoch = 0
    SSIM_best = (start_epoch, 0.)
    LPIPS_best = (start_epoch, 1.)
    

    if opt.weight != '':
        # 加载预训练权重
        ckpt = torch.load(opt.weight)
        pix_G.load_state_dict(ckpt['G_model'])
        pix_D.load_state_dict(ckpt['D_model'])
        start_epoch = ckpt['epoch'] + 1

        # 加载训练日志，获取最佳参数
        val_loss_ssim = []
        ssim_log_file = glob.glob('./train_logs/val_loss_ssim/events.out.tfevents.*')
        for logfile in ssim_log_file:
            ea = event_accumulator.EventAccumulator(logfile) 
            ea.Reload()
            if 'val' in ea.scalars.Keys():
                val_loss_ssim += ea.scalars.Items('val')
        if not len(val_loss_ssim) == 0:
            val_loss_ssim.sort(key=lambda x: x.value)
            SSIM_best = (val_loss_ssim[-1].step + 1, val_loss_ssim[-1].value)

        val_loss_lpips = []
        lpips_log_file = glob.glob('./train_logs/val_loss_lpips/events.out.tfevents.*')
        for logfile in lpips_log_file:
            ea = event_accumulator.EventAccumulator(logfile) 
            ea.Reload()
            if 'val' in ea.scalars.Keys():
                val_loss_lpips += ea.scalars.Items('val')
        if not len(val_loss_lpips) == 0:
            val_loss_lpips.sort(key=lambda x: x.value)
            LPIPS_best = (val_loss_lpips[-1].step + 1, val_loss_lpips[-1].value)

    # 定义梯度放大器、优化器
    scaler_G = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_D = torch.cuda.amp.GradScaler(enabled=use_amp)
    optim_G = bnb.optim.AdamW(pix_G.parameters(), lr=0.00001, betas=(0.5, 0.999), weight_decay=0)
    optim_D = bnb.optim.AdamW(pix_D.parameters(), lr=0.00001, betas=(0.5, 0.999), weight_decay=0)

    # 定义损失函数
    loss_mse = nn.MSELoss()
    loss_ssim = SSIM(data_range=1, size_average=True)
    loss_lpips = LPIPS(net='vgg', verbose=False).to(device)

    # 初始化tensorboard记录
    writer = SummaryWriter('train_logs')

    # 加载数据集
    train_imglist, val_imglist = split_data(data_path)
    train_datasets = CreateDatasets(train_imglist, img_size)
    val_datasets = CreateDatasets(val_imglist, img_size)

    train_loader = DataLoader(dataset=train_datasets, batch_size=train_batch_size, shuffle=True,
                              drop_last=True, num_workers=opt.numworker, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(dataset=val_datasets, batch_size=val_batch_size, shuffle=True,
                            drop_last=True, num_workers=opt.numworker, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    print("train dataset size:", len(train_datasets))
    print("val dataset size:", len(val_datasets))


    # 开始训练
    for epoch in range(start_epoch, epochs):
        train_loss_mG, train_loss_mD, train_mSSIM, train_mLPIPS = train_one_epoch(
            G=pix_G,
            D=pix_D,
            train_loader=train_loader,
            writer=writer,
            optim_G=optim_G,
            optim_D=optim_D,
            scaler_G=scaler_G,
            scaler_D=scaler_D,
            GAN_loss=loss_mse,
            lpips_loss=loss_lpips,
            ssim_loss=loss_ssim,
            image_size=img_size,
            batch_size=train_batch_size,
            plot_every=print_every,
            device=device,
            epoch=epoch,
            epochs=epochs,
            use_amp=use_amp,
            accumulation_steps=accumulation_steps  # Pass the accumulation steps
        )

        if True in np.isnan([train_loss_mG, train_loss_mD, train_mSSIM, train_mLPIPS]):
            raise RuntimeError('Train losses have nan value')

        val_loss_mG, val_loss_mD, val_mSSIM, val_mLPIPS, best_image = val_one_epoch(
            G=pix_G,
            D=pix_D,
            val_loader=val_loader,
            writer=writer,
            GAN_loss=loss_mse,
            ssim_loss=loss_ssim,
            lpips_loss=loss_lpips,
            image_size=img_size,
            batch_size=val_batch_size,
            device=device,
            epoch=epoch,
            epochs=epochs
        )

        writer.add_scalars(main_tag='train', tag_scalar_dict={
            'loss_G': train_loss_mG,
            'loss_D': train_loss_mD,
            'loss_ssim': train_mSSIM,
            'loss_lpips': train_mLPIPS
        }, global_step=epoch)

        writer.add_scalars(main_tag='val', tag_scalar_dict={
            'loss_G': val_loss_mG,
            'loss_D': val_loss_mD,
            'loss_ssim': val_mSSIM,
            'loss_lpips': val_mLPIPS
        }, global_step=epoch)

        writer.add_images(tag='val_results_best', img_tensor=best_image, global_step=epoch)

        _SSIM_stat = (epoch + 1, val_mSSIM)
        _LPIPS_stat = (epoch + 1, val_mLPIPS)
        print(f'Epoch count: {epoch + 1}, Train SSIM = ' + str(train_mSSIM) + ', Val SSIM = ' + str(val_mSSIM))
        print(f'Epoch count: {epoch + 1}, Train LPIPS = ' + str(train_mLPIPS) + ', Val LPIPS = ' + str(val_mLPIPS))

        # 保存模型
        torch.save({
            'G_model': pix_G.state_dict(),
            'D_model': pix_D.state_dict(),
            'epoch': epoch
        }, weight_path + f'/aivis_13channels_epoch{epoch}.pth')
        torch.save({
            'G_model': pix_G.state_dict(),
            'D_model': pix_D.state_dict(),
            'epoch': epoch
        }, weight_path + '/load.pth')
        print(f'Epoch count: {epoch + 1}, Model saved! Time: {datetime.now()}')

        if _SSIM_stat[1] > SSIM_best[1] and _LPIPS_stat[1] < LPIPS_best[1]:
            SSIM_best = _SSIM_stat
            LPIPS_best = _LPIPS_stat
            torch.save({
                'G_model': pix_G.state_dict(),
                'D_model': pix_D.state_dict(),
                'epoch': epoch
            }, weight_path + '/aivis_13channels_best.pth')
            print(f'Epoch count: {epoch + 1}, Best Model updated!')

    print("Best Val SSIM = " + str(SSIM_best[1]) + " & LPIPS = " + str(LPIPS_best[1]) + " at epoch " + str(SSIM_best[0]))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch', type=int, default=10)
    parse.add_argument('--epoch', type=int, default=110)
    parse.add_argument('--imgsize', type=int, default=512)
    parse.add_argument('--dataPath', type=str, default="./datasets/Band13_15_az", help='data root path')
    parse.add_argument('--weight', type=str, default="./weights/load.pth", help='load pre train weight')
    parse.add_argument('--savePath', type=str, default='./weights', help='weight save path')
    parse.add_argument('--numworker', type=int, default=4)
    parse.add_argument('--every', type=int, default=0, help='plot train result every * iters')
    parse.add_argument('--amp', default=False, action='store_true', help='Experimental option of using automatic mixed precision training')
    parse.add_argument('--grad-accu', type=int, default=1, help='Number of gradient accumulation steps')
    opt = parse.parse_args()
    print(opt)
    main(opt)
