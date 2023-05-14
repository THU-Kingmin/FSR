import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from dataset.dataset import  TrainDataset,TestDataset
import matplotlib.pyplot as plt
from collections import OrderedDict

def load_dataset(rank,hr_dir, lr_dir, patch, scale, batch, n_wks):
    train_dataset = TrainDataset(hr_dir, lr_dir, patch=patch, scale=scale)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True,rank=rank)
    train_dataset_loader = DataLoader(train_dataset,sampler=train_sampler, batch_size=batch,drop_last=True, num_workers=n_wks,pin_memory=True)
    # train_dataset_loader = DataLoader(train_dataset, batch_size=batch, num_workers=n_wks, shuffle=True, drop_last=True,pin_memory=True)
    print('train_dataset_loader : ',len(train_dataset_loader))
    return train_dataset_loader

def load_test_dataset(hr_dir,lr_dir, patch, scale, batch, n_wks):
    test_dataset = TestDataset(hr_dir,lr_dir, patch=patch, scale=scale)
    test_dataset_loader = DataLoader(test_dataset, batch_size=batch, num_workers=n_wks, shuffle=False, drop_last=True)
    print('test_dataset_loader : ',len(test_dataset_loader))
    return test_dataset_loader

def save_checkpoint(model, optimizer,scheduler,epoch,args):
    model_ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    patch, batch, scale, whichModule = args.patch, args.batch, args.scale, args.whichModule
    ckp_dir = args.ckp_dir
    model_path = os.path.join(ckp_dir,'FSR_{}_P{}_X{}_E{}.pth.tar'.format(whichModule,patch,scale,epoch))
    torch.save(model_ckpt, model_path)

def load_checkpoint(model, optimizer,scheduler,args):
    patch, batch, scale, whichModule, sta_e = args.patch, args.batch, args.scale, args.whichModule, args.sta_e
    ckp_dir = args.ckp_dir
    model_path = os.path.join(ckp_dir,'FSR_{}_P{}_X{}_E{}.pth.tar'.format(whichModule,patch,scale,sta_e))
    model_ckpt = torch.load(model_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in model_ckpt['model'].items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    # model.load_state_dict(model_ckpt['model'])
    model.load_state_dict(load_net_clean)
    optimizer.load_state_dict(model_ckpt['optimizer'])
    scheduler.load_state_dict(model_ckpt['scheduler'])
    return model, optimizer,scheduler

def printPara(model):
    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    print("parameters : ",num_parameters//1000,"k")

def save_PSNR(epochs,psnr,args,stage = 'Valid'):
    plt.plot(epochs, psnr[0], epochs, psnr[1], epochs, psnr[2], epochs, psnr[3], epochs, psnr[4])
    plt.legend(['a','h','v','d','sr'])
    plt.title('{} PSNR and Epoch'.format(stage))
    patch, batch, scale, whichModule = args.patch, args.batch, args.scale, args.whichModule
    filename = os.path.join(args.log_dir,'FSR_PSNR_{}_{}_P{}_X{}_E{}.png'.format(stage,whichModule,patch,scale,epochs[-1]))
    plt.savefig(filename)
    plt.close()

def save_loss(epochs,loss,args,stage = 'Train'):
    plt.plot(epochs, loss[0],epochs, loss[1],epochs, loss[2],epochs, loss[3],epochs, loss[4],epochs, loss[5])
    plt.legend(['a','h','v','d','sr','total'])
    plt.title('{} Loss and Epoch'.format(stage))
    patch, batch, scale, whichModule = args.patch, args.batch, args.scale, args.whichModule
    filename = os.path.join(args.log_dir,'FSR_Loss_{}_{}_P{}_X{}_E{}.png'.format(stage,whichModule,patch,scale,epochs[-1]))
    plt.savefig(filename)
    plt.close()

def printBestPSNRandEpoch(bestPSNR,BestEpoch,newPSNR,newEpoch,stage = 'Valid'):
    print('{} : '.format(stage))
    for i in range(len(bestPSNR)):
        if newPSNR[i] > bestPSNR[i] :
            bestPSNR[i] = newPSNR[i]
            BestEpoch[i] = newEpoch[i]
    
    print('Best PSNR : ',bestPSNR)
    print('Best Epoch : ',BestEpoch)
    print('new PSNR : ',newPSNR)
    print('New Epoch : ',newEpoch)
    return bestPSNR, BestEpoch
