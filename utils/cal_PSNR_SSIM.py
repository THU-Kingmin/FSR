import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import cv2

def cal_psnr(img1, img2):
    '''
    img1 and img2 have range [0, 1], C * H * W, numpy daarray
    '''
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    img1 = im2double(img1)
    img2 = im2double(img2)
    psnr = PSNR(img1, img2, data_range=1)
    return psnr

def cal_psnr_y(img1, img2):
    '''
    img1 and img2 have range [0, 1], C * H * W, numpy daarray
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    img1 = rgb2ycbcr(img1)
    img2 = rgb2ycbcr(img2)
    psnr = PSNR(img1, img2,data_range=1)
    return psnr

def cal_ssim(img1, img2,testRGB=True):
    '''
    img1 and img2 have range [0, 1], C * H * W, numpy daarray
    '''
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    img1 = im2double(img1)
    img2 = im2double(img2)
    if testRGB:
        ssim = SSIM(img1, img2)
    else:
        y1 = rgb2ycbcr(img1)
        y2 = rgb2ycbcr(img2)
        ssim = SSIM(y1, y2)
    return ssim

def rgb2ycbcr(img,only_y=True):
    '''
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ssim_classSR(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()