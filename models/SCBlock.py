import torch.nn as nn
import torch
import torch.nn.functional as F

class SCBlock(torch.nn.Module):
    def __init__(self, input_channels):
        super(SCBlock, self).__init__()
        self.feature_tran1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=1, stride=1),
            nn.ReLU())
        self.feature_tran3 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.feature_tran5 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.mp1 = nn.MaxPool2d(2, stride=2)
        self.mp3 = nn.MaxPool2d(2, stride=2)
        self.mp5 = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        fea1 = self.feature_tran1(x)
        out1 = self.mp1(fea1)
        fea3 = self.feature_tran3(x)
        out3 = self.mp3(fea3)
        fea5 = self.feature_tran5(x)
        out5 = self.mp5(fea5)
        return out1 + out3 + out5

  
if __name__ == '__main__':
    model = SCBlock(3)
    x = torch.randn(14, 3, 64, 64)
    out = model(x)
    print(out.shape)