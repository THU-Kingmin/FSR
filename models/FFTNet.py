import torch.nn as nn
import torch

class FFTNet(torch.nn.Module):
    '''
    input : (N * C * H * W) 
    output: (N * C' * H * W)
    '''
    def __init__(self, in_channels, out_channels):
        super(FFTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReLU())
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

if __name__ == '__main__':
    model = FFTNet(9,3)
    x = torch.randn(2, 9, 64, 64)
    out = model(x)
    print(out.shape)
