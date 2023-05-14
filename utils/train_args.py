import argparse
import argparse

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = '0,1,2,3', help ='--device')
    parser.add_argument('--local_rank', default = 0, help ='--rank')
    parser.add_argument('--hr_dir', default = '../dataset64/train/X4/HR/', help ='--hr_dir')
    parser.add_argument('--lr_dir', default = '../dataset64/train/X4/LR/', help ='--lr_dir')
    parser.add_argument('--ckp_dir', default = './checkpoint/', help ='--ckp_dir')
    parser.add_argument('--log_dir', default = './log/', help ='--log_dir')
    parser.add_argument('--patch' , default = 64, type = int, help = '--patch')
    parser.add_argument('--batch' , default = 16,  type = int, help = '--batch')
    parser.add_argument('--in_ch' , default = 3 , type = int, help = '--input channels')
    parser.add_argument('--scale' , default = 4 , type = int, help = '--scale')
    parser.add_argument('--n_wks' , default = 2 , type = int, help = '--n_wks')
    parser.add_argument('--whichModule', choices=['fsrcnn','carn','srresnet','rcan'], default = 'fsrcnn', help ='--whichModule')
    parser.add_argument('--wl' , default = 10, type = int, help = '--loss weight for low frequency')
    parser.add_argument('--wh' , default = 1, type = int, help = '--loss weight for high frequency')
    parser.add_argument('--wr' , default = 1, type = float, help = '--loss weight for reconstruct full image')
    parser.add_argument("--beta", type = float, default=20.0)
    parser.add_argument('--rec_type', default = 'l2', help ='--rec_type')
    parser.add_argument('--fre_type', default = 'char', help ='--fre_type')
    parser.add_argument("--clip", type = float, default=8.0)
    parser.add_argument("--grad", type = bool, default=False)
    
    parser.add_argument('--epochs', default = 4000 , type = int, help = '--epochs')
    parser.add_argument('--sta_e' , default = 0 , type = int, help = '--sta_e')
    parser.add_argument('--intval', default = 5 , type = int, help = '--intval')
    parser.add_argument('--isload', default = 0 , type = int, help = '--isload')
    parser.add_argument('--lr' , default = 2e-4, type = float, help = '--learning rate')
    parser.add_argument('--T_max' , default = 6000, type = int, help = '--Period of CosineAnnealingLR')
    parser.add_argument('--eta_min' , default = 1e-7, type = float, help = '--min learning rate')

    parser.add_argument('--hr_dir_test', default = '../../code/test2k/HR/', help ='--hr_dir_test')
    parser.add_argument('--lr_dir_test', default = '../../code/test2k/LR/', help ='--lr_dir_test')
    parser.add_argument('--patch_test' , default = 64, type = int, help = '--test patch')
    parser.add_argument('--batch_test' , default = 1,  type = int, help = '--batch')
    parser.add_argument('--n_wks_test' , default = 4 , type = int, help = '--n_wks')

    args = parser.parse_args()
    return args
