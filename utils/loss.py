import torch
import torch.nn as nn

class Charbonnier_loss(nn.Module):
    def __init__(self):
        super(Charbonnier_loss, self).__init__()
        self.eps = 1e-6
 
    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class Reconstruct_loss(nn.Module):
    def __init__(self, losstype='l2'):
        super(Reconstruct_loss, self).__init__()
        if losstype == 'l2':
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.loss_fn = torch.nn.L1Loss()

    def forward(self, x, y):
       return self.loss_fn(x,y)

if __name__ == '__main__':
    loss_ch = Charbonnier_loss()
    loss_l2 = Reconstruct_loss('l2')
    SR  = torch.randn(2, 3, 64, 64)
    HR  = torch.randn(2, 3, 64, 64)
    print(loss_ch(SR,HR))
    print(loss_l2(SR,HR))
