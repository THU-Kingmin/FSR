import torch
import torch.nn as nn
from torch.nn import Softmax
from models.FFTNet import FFTNet

def INF(B,H,W):
    if torch.cuda.is_available():
        return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
    else :
        return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class TABlock(nn.Module):
    '''
    input : (N * C * H * W) q:A k,v:[H V D],k==v
    output: (N * C * H * W) A'
    '''
    def __init__(self, in_channels):
        super(TABlock,self).__init__()
        self.q_conv = FFTNet(in_channels, 3)
        self.k_conv = FFTNet(in_channels*3, 3)
        self.v_conv = FFTNet(in_channels*3, 3)
        self.s_conv = FFTNet(in_channels, 3)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, query, key, spatial):
        m_batchsize, _, height, width = query.size()
        proj_query = self.q_conv(query)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.k_conv(key)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.v_conv(key)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) #+ self.s_conv(spatial)


if __name__ == '__main__':
    model = TABlock(3)
    low_fre = torch.randn(2, 3, 64, 64)
    high_fre = torch.randn(2, 9, 64, 64)
    out = model(low_fre,high_fre)
    print(out.shape)
