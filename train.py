import os
import torch
from utils.train_args import train_args
from models.FSR import FSR
from models.HaarWavelet import HaarWavelet
from utils.loss import Charbonnier_loss, Reconstruct_loss
from torch.optim import lr_scheduler as lr_scheduler
import torch.optim as optim
from utils.trainInfo import *
from tqdm import tqdm
from datetime import datetime
from test import test

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

def train(rank, world_size):
    args = train_args()
    dist.init_process_group('NCCL', rank=rank, world_size=world_size)
    device = torch.device('cuda',rank)
    torch.cuda.set_device(device)
    # parameters
    hr_dir, lr_dir, patch, batch = args.hr_dir, args.lr_dir, args.patch, args.batch
    in_channels, scale, n_wks = args.in_ch, args.scale, args.n_wks
    whichModule, wl, wh, wr = args.whichModule, args.wl, args.wh, args.wr
    epochs, sta_e, intval, isload = args.epochs, args.sta_e, args.intval, args.isload
    learning_rate, T_max, eta_min = args.lr, args.T_max, args.eta_min
    rec_type, beta, clip = args.rec_type,args.beta, args.clip
    grad = args.grad
    torch.backends.cudnn.benchmark = True

    # load
    train_dataset_loader = load_dataset(rank,hr_dir, lr_dir, patch, scale, batch, n_wks)
    # init
    model, HaarWL = FSR(in_channels, whichModule,HaarGrad=grad).to(device), HaarWavelet(in_channels).to(device)
    loss_sr, loss_fre = Reconstruct_loss(losstype=rec_type),  Reconstruct_loss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate, betas=(0.9, 0.999))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    if isload != 0 and rank == 0:
        model, optimizer,scheduler = load_checkpoint(model, optimizer,scheduler,args) 
    model= DDP(model, device_ids=[rank], find_unused_parameters=True)
    

    # log
    if rank == 0 :
        printPara(model)
    # train
    e_losses, e_epochs, e_psnr1, e_psnr2, fig_epochs, fig_losses = [[],[],[],[],[],[]], [], [[],[],[],[],[]], [[],[],[],[],[]], [], [[],[],[],[],[],[]]
    bestPSNR1, BestEpoch1, bestPSNR2, BestEpoch2 = [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]
    
    for epoch in range(sta_e,epochs):
        losses=[[],[],[],[],[],[]]
        wr =  min(1 + beta * epoch/epochs,wl)
        with tqdm(
                iterable=train_dataset_loader,
                bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',
                ncols=100
        ) as t:
            start_time = datetime.now()
            for i,inputs in enumerate(train_dataset_loader) :
                hr, lr = inputs[0].to(device), inputs[1].to(device)
                hr_ahvd = HaarWL(hr)
                hr_a, hr_h, hr_v, hr_d = hr_ahvd.narrow(1, 0, in_channels), hr_ahvd.narrow(1, in_channels, in_channels),\
                    hr_ahvd.narrow(1, in_channels*2, in_channels), hr_ahvd.narrow(1, in_channels*3, in_channels)
                
                optimizer.zero_grad()
                sr_a, sr_h, sr_v, sr_d, sr = model(lr)
                loss_a, loss_h, loss_v, loss_d = loss_fre(sr_a, hr_a), loss_fre(sr_h, hr_h), loss_fre(sr_v, hr_v), loss_fre(sr_d, hr_d)
                loss_rec = loss_sr(sr,hr)
                #wr =  0.1 + 100*epoch/epochs * loss_a.item()/loss_rec.item()
                total_loss = wl * loss_a + wh * loss_h + wh * loss_v + wh * loss_d + wr * loss_rec
                losses[0].append(loss_a.item()), losses[1].append(loss_h.item()), losses[2].append(loss_v.item()),\
                    losses[3].append(loss_d.item()), losses[4].append(loss_rec.item()), losses[5].append(total_loss.item())

                total_loss.backward()
                if whichModule == 'carn' or whichModule == 'rcan':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                
                delta_time = datetime.now() - start_time
                t.set_description_str(f"\33[36m【E {epoch + 1:04d}】")
                t.set_postfix_str(f"la={sum(losses[0]) / len(losses[0]):.6f}，sr={sum(losses[4]) / len(losses[4]):.6f}， 时长：{delta_time}\33[0m")
                t.update()
            scheduler.step()
        fig_losses[0].append(sum(losses[0])/len(losses[0])),fig_losses[1].append(sum(losses[1])/len(losses[1])),fig_losses[2].append(sum(losses[2])/len(losses[2])),\
            fig_losses[3].append(sum(losses[3])/len(losses[3])),fig_losses[4].append(sum(losses[4])/len(losses[4])),fig_losses[5].append(sum(losses[5])/len(losses[5]))
        fig_epochs.append(epoch+1)
        #save model
        if (epoch+1) % intval == 0 and rank == 0:
            save_checkpoint(model, optimizer, scheduler, epoch+1, args)
            # e_losses[0].append(sum(losses[0])/len(losses[0])), e_losses[1].append(sum(losses[1])/len(losses[1])), e_losses[2].append(sum(losses[2])/len(losses[2])),\
            #     e_losses[3].append(sum(losses[3])/len(losses[3])), e_losses[4].append(sum(losses[4])/len(losses[4])), e_losses[5].append(sum(losses[5])/len(losses[5]))
            # e_epochs.append(epoch+1)

            # newPSNR2, newSSIM2 = test(args,epoch+1)
            # e_psnr2[0].append(newPSNR2[0]), e_psnr2[1].append(newPSNR2[1]), e_psnr2[2].append(newPSNR2[2]), \
            #     e_psnr2[3].append(newPSNR2[3]), e_psnr2[4].append(newPSNR2[4])
            # printBestPSNRandEpoch(bestPSNR2, BestEpoch2, newPSNR2, [epoch+1,epoch+1,epoch+1,epoch+1,epoch+1],stage='Test')

            # save_loss(fig_epochs,fig_losses,args,stage='Train')
    
    # if rank == 0:
    #     save_loss(e_epochs,e_losses,args,stage='Train')
    #     save_PSNR(e_epochs,e_psnr2,args,stage='Test')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'
    # train(args)
    world_size = 4
    mp.spawn(train,
        args=(world_size,),
        nprocs=world_size,
        join=True)