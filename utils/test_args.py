import argparse

def test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = '0,1,2,3', help ='--device')
    parser.add_argument('--hr_dir_test', default = '../../code/test2k/HR/', help ='--hr_dir_test')
    parser.add_argument('--lr_dir_test', default = '../../code/test2k/LR/', help ='--lr_dir_test')
    parser.add_argument('--ckp_dir', default = './checkpoint/', help ='--ckp_dir')
    parser.add_argument('--log_dir', default = './log/', help ='--log_dir')
    parser.add_argument('--patch' , default = 64, type = int, help = '--train patch')
    parser.add_argument('--patch_test' , default = 64, type = int, help = '--test patch')
    parser.add_argument('--batch_test' , default = 1,  type = int, help = '--test batch')
    parser.add_argument('--in_ch' , default = 3 , type = int, help = '--input channels')
    parser.add_argument('--scale' , default = 4 , type = int, help = '--scale')
    parser.add_argument('--n_wks_test' , default = 8 , type = int, help = '--n_wks')
    parser.add_argument('--whichModule', choices = ['fsrcnn','carn','srresnet','rcan'], default = 'fsrcnn', help ='--whichModule')
    parser.add_argument('--epochs', default = 1000 , type = int, help = '--epochs')
    args = parser.parse_args()
    return args