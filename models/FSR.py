import functools
import torch.nn as nn
import models.arch_util as arch_util
import torch.nn.functional as F
import torch
from torch.nn import Softmax
import math
from models.HaarWavelet import HaarWavelet
from models.SRmodule import FSRCNN_net,CARN_M,MSRResNet,RCAN
from models.SCBlock import SCBlock
from models.TABlock import TABlock

class FSR(torch.nn.Module):
    '''
    input : (N * C * H * W) 
    output: 
    sr_a : (N * C * (H*scale//2) * (W*scale//2)) 
    sr_h : (N * C * (H*scale//2) * (W*scale//2)) 
    sr_v : (N * C * (H*scale//2) * (W*scale//2)) 
    sr_d : (N * C * (H*scale//2) * (W*scale//2)) 
    sr   : (N * C * (H*scale) * (W*scale)) 
    '''
    def __init__(self, in_channels,whichModule,upscale=4,HaarGrad=False):
        super(FSR, self).__init__()
        self.in_channels = in_channels
        self.whichModule = whichModule
        self.scale = upscale
        self.Spatial = SCBlock(in_channels)
        self.haarWavelet1 = HaarWavelet(in_channels,grad=HaarGrad)
        self.haarWavelet2 = HaarWavelet(in_channels,grad=False)
        self.haarWavelet3 = HaarWavelet(in_channels,grad=HaarGrad)
        self.haarWavelet4 = HaarWavelet(in_channels,grad=False)
        self.freAttention1 = TABlock(in_channels)
        self.freAttention2 = TABlock(in_channels)
        self.freAttention3 = TABlock(in_channels)
        self.freAttention4 = TABlock(in_channels)

        rgb_range = 1
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = arch_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = arch_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        if whichModule == 'fsrcnn' :
            self.SRNet_a = FSRCNN_net(in_channels*6,in_channels, upscale=upscale, d=64, s=12, m=4)
            self.SRNet_h = FSRCNN_net(in_channels*6,in_channels, upscale=upscale, d=16, s=12, m=4)
            self.SRNet_v = FSRCNN_net(in_channels*6,in_channels, upscale=upscale, d=16, s=12, m=4)
            self.SRNet_d = FSRCNN_net(in_channels*6,in_channels, upscale=upscale, d=14, s=12, m=4)
        if whichModule == 'carn' :
            self.SRNet_a = CARN_M(in_channels*6,in_channels,64,scale=upscale)
            self.SRNet_h = CARN_M(in_channels*6,in_channels,44,scale=upscale)
            self.SRNet_v = CARN_M(in_channels*6,in_channels,44,scale=upscale)
            self.SRNet_d = CARN_M(in_channels*6,in_channels,44,scale=upscale)
        if whichModule == 'srresnet' :
            self.SRNet_a = MSRResNet(in_nc=in_channels*6, out_nc=in_channels, nf=64, nb=16, upscale=4)
            self.SRNet_h = MSRResNet(in_nc=in_channels*6, out_nc=in_channels, nf=52, nb=12, upscale=4)
            self.SRNet_v = MSRResNet(in_nc=in_channels*6, out_nc=in_channels, nf=52, nb=12, upscale=4)
            self.SRNet_d = MSRResNet(in_nc=in_channels*6, out_nc=in_channels, nf=40, nb=12, upscale=4)
        if whichModule == 'rcan' :
            self.SRNet_a = RCAN(n_resgroups=10, n_resblocks=20, n_feats=64, res_scale=1, n_colors=in_channels*6, rgb_range=1,scale=4, reduction=16)
            self.SRNet_h = RCAN(n_resgroups=10, n_resblocks=20, n_feats=48, res_scale=1, n_colors=in_channels*6, rgb_range=1,scale=4, reduction=16)
            self.SRNet_v = RCAN(n_resgroups=10, n_resblocks=20, n_feats=48, res_scale=1, n_colors=in_channels*6, rgb_range=1,scale=4, reduction=16)
            self.SRNet_d = RCAN(n_resgroups=10, n_resblocks=20, n_feats=32, res_scale=1, n_colors=in_channels*6, rgb_range=1,scale=4, reduction=16)

    def forward(self, x):
        if self.whichModule == 'fsrcnn' :
            out = self.forward_SR(x)
        if self.whichModule == 'carn' :
            out = self.forward_SR(x)
        if self.whichModule == 'srresnet' :
            out = self.forward_SR(x)
        if self.whichModule == 'rcan' :
            out = self.forward_SR(x)
        return out

    def forward_SR(self,x):
        x = self.sub_mean(x)
        
        out = self.haarWavelet1(x,rev=False)
        spa = self.Spatial(x)

        a = out.narrow(1, 0, self.in_channels)
        h = out.narrow(1, self.in_channels, self.in_channels)
        v = out.narrow(1, self.in_channels*2, self.in_channels)
        d = out.narrow(1, self.in_channels*3, self.in_channels)
        ahv = torch.cat([a,h,v],1)
        ahd = torch.cat([a,h,d],1)
        avd = torch.cat([a,v,d],1)
        hvd = torch.cat([h,v,d],1)

        atn_a = self.freAttention1(a,hvd,spa)
        atn_h = self.freAttention2(h,avd,spa)
        atn_v = self.freAttention3(v,ahd,spa)
        atn_d = self.freAttention4(d,ahv,spa)

        fea_a = torch.cat([out,atn_a,spa],1)
        fea_h = torch.cat([out,atn_h,spa],1) 
        fea_v = torch.cat([out,atn_v,spa],1) 
        fea_d = torch.cat([out,atn_d,spa],1) 
       
        sr_a = self.SRNet_a(fea_a)
        sr_h = self.SRNet_h(fea_h)
        sr_v = self.SRNet_v(fea_v)
        sr_d = self.SRNet_d(fea_d)


        x2 = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        out2 = self.haarWavelet2(x2,rev=False)
        a2 = out2.narrow(1, 0, self.in_channels)
        h2 = out2.narrow(1, self.in_channels, self.in_channels)
        v2 = out2.narrow(1, self.in_channels*2, self.in_channels)
        d2 = out2.narrow(1, self.in_channels*3, self.in_channels)

        sr_a = sr_a + a2
        sr_h = sr_h + h2
        sr_v = sr_v + v2
        sr_d = sr_d + d2

        sr_ahvd = torch.cat([sr_a,sr_h,sr_v,sr_d],1)
        sr = self.haarWavelet3(sr_ahvd,rev=True)

        sr = self.add_mean(sr)

        sr2 = self.haarWavelet4(sr,rev=False)
        sr_a2 = sr2.narrow(1, 0, self.in_channels)
        sr_h2 = sr2.narrow(1, self.in_channels, self.in_channels)
        sr_v2 = sr2.narrow(1, self.in_channels*2, self.in_channels)
        sr_d2 = sr2.narrow(1, self.in_channels*3, self.in_channels)

        return sr_a2, sr_h2, sr_v2, sr_d2, sr
