import os
import numpy as np
import torch
from utils.test_args import test_args
from models.HaarWavelet import HaarWavelet
from models.FSR import FSR
from utils.get_path import get_model_path
from utils.cal_PSNR_SSIM import cal_psnr,cal_ssim
from utils.trainInfo import load_test_dataset
# import cv2

def batch2full(batch_img,w,h,numw,numh,step=116,patch=128,in_channels=3):
    index=0
    full_image = np.zeros((h, w, in_channels), 'float32')
    for i in range(numh):
        for j in range(numw):
            full_image[i*step:i*step+patch,j*step:j*step+patch,:]+=batch_img[index]
            index+=1
    full_image=full_image.astype('float32')

    for j in range(1,numw):
        full_image[:,j*step:j*step+patch-step,:]/=2

    for i in range(1,numh):
        full_image[i*step:i*step+patch-step,:,:]/=2
    return full_image
    
def test(args,epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load
    hr_dir,lr_dir, patch_test, batch = args.hr_dir_test, args.lr_dir_test, args.patch_test, args.batch_test
    in_channels, scale, n_wks = args.in_ch, args.scale, args.n_wks_test
    whichModule = args.whichModule

    test_dataset_loader = load_test_dataset(hr_dir, lr_dir, patch_test, scale, batch, n_wks)
    model, HaarWL = FSR(in_channels, whichModule).to(device), HaarWavelet(in_channels)
    model= torch.nn.DataParallel(model)
    model_path = get_model_path(args, epoch)
    print(model_path)
    model_ckpt = torch.load(model_path)
    model.load_state_dict(model_ckpt['model'])

    # train
    with torch.no_grad():
        psnr, ssim = [[],[],[],[],[]], []
        for i,inputs in enumerate(test_dataset_loader) :
            print('test',i)
            hr, lr,numw, numh, hw1, hh1, hw, hh = inputs[0].squeeze(0), inputs[1].squeeze(0).to(device), inputs[2], inputs[3],\
                inputs[4], inputs[5], inputs[6], inputs[7]
            f_hw1, f_hh1 = torch.div(hw1, 2, rounding_mode='floor'), torch.div(hh1, 2, rounding_mode='floor')
            # f_hw, f_hh = torch.div(hw, 2, rounding_mode='floor'), torch.div(hh, 2, rounding_mode='floor')
            hr_ahvd = HaarWL(hr)
            hr_a, hr_h, hr_v, hr_d = hr_ahvd.narrow(1, 0, in_channels), hr_ahvd.narrow(1, in_channels, in_channels),\
                hr_ahvd.narrow(1, in_channels*2, in_channels), hr_ahvd.narrow(1, in_channels*3, in_channels)
            
            sr_a, sr_h, sr_v, sr_d, sr = model(lr)
            sr_a, sr_h, sr_v, sr_d, sr = sr_a.to("cpu"), sr_h.to("cpu"), sr_v.to("cpu"), sr_d.to("cpu"), sr.to("cpu")

            hr = hr.mul(255).clamp(0, 255).byte().permute(0,2,3,1).numpy()
            hr_a = hr_a.mul(255).clamp(0, 255).byte().permute(0,2,3,1).numpy()
            hr_h = hr_h.mul(255).clamp(0, 255).byte().permute(0,2,3,1).numpy()
            hr_v = hr_v.mul(255).clamp(0, 255).byte().permute(0,2,3,1).numpy()
            hr_d = hr_d.mul(255).clamp(0, 255).byte().permute(0,2,3,1).numpy()
            sr = sr.mul(255).clamp(0, 255).byte().permute(0,2,3,1).numpy()
            sr_a = sr_a.mul(255).clamp(0, 255).byte().permute(0,2,3,1).numpy()
            sr_h = sr_h.mul(255).clamp(0, 255).byte().permute(0,2,3,1).numpy()
            sr_v = sr_v.mul(255).clamp(0, 255).byte().permute(0,2,3,1).numpy()
            sr_d = sr_d.mul(255).clamp(0, 255).byte().permute(0,2,3,1).numpy()
            
            sr_a, sr_h, sr_v, sr_d, sr = batch2full(sr_a,f_hw1,f_hh1,numw,numh), batch2full(sr_h,f_hw1,f_hh1,numw,numh), batch2full(sr_v,f_hw1,f_hh1,numw,numh), \
                batch2full(sr_d,f_hw1,f_hh1,numw,numh), batch2full(sr,hw1,hh1,numw,numh,step=232,patch=256)
            hr_a, hr_h, hr_v, hr_d, hr = batch2full(hr_a,f_hw1,f_hh1,numw,numh), batch2full(hr_h,f_hw1,f_hh1,numw,numh), batch2full(hr_v,f_hw1,f_hh1,numw,numh), \
                batch2full(hr_d,f_hw1,f_hh1,numw,numh), batch2full(hr,hw1,hh1,numw,numh,step=232,patch=256)
            
            psnr[0].append(cal_psnr(sr_a,hr_a))
            psnr[1].append(cal_psnr(sr_h,hr_h))
            psnr[2].append(cal_psnr(sr_v,hr_v))
            psnr[3].append(cal_psnr(sr_d,hr_d))
            psnr[4].append(cal_psnr(sr,hr))
            ssim.append(cal_ssim(sr,hr,testRGB=False))
            hr_a, hr_h, hr_v, hr_d, hr = hr_a[:, :, [2, 1, 0]], hr_h[:, :, [2, 1, 0]], hr_v[:, :, [2, 1, 0]], hr_d[:, :, [2, 1, 0]], hr[:, :, [2, 1, 0]]
            sr_a, sr_h, sr_v, sr_d, sr = sr_a[:, :, [2, 1, 0]], sr_h[:, :, [2, 1, 0]], sr_v[:, :, [2, 1, 0]], sr_d[:, :, [2, 1, 0]], sr[:, :, [2, 1, 0]]
            
            # cv2.imwrite('./images/{}_hr.png'.format(i), hr*255)
            # cv2.imwrite('./images/{}_sr.png'.format(i), sr*255)
        new_psnr = [sum(psnr[0])/len(psnr[0]),sum(psnr[1])/len(psnr[1]),sum(psnr[2])/len(psnr[2]),\
            sum(psnr[3])/len(psnr[3]),sum(psnr[4])/len(psnr[4])]
        new_ssim_y = sum(ssim)/len(ssim)
        return new_psnr, new_ssim_y

if __name__ == '__main__':
    args=test_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    psnr, ssim = test(args,args.epochs)
    print('test : ')
    print(psnr,ssim)