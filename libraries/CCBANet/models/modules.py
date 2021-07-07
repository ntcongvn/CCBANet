import torch
from torch import nn
from torch.nn import functional as F


class ASM(nn.Module):
    def __init__(self, in_channels, all_channels):
        super(ASM, self).__init__()
        self.non_local = NonLocalBlock(in_channels)
        self.selayer = SELayer(all_channels)

    def forward(self, bam, fuse, ccm):
        fuse = self.non_local(fuse)
        fuse = torch.cat([bam, fuse, ccm], dim=1)
        fuse = self.selayer(fuse)

        return fuse

"""
Squeeze and Excitation Layer

https://arxiv.org/abs/1709.01507

"""

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


"""
Non Local Block

https://arxiv.org/abs/1711.07971
"""


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


#Cascading Context Module
class CCM(nn.Module):
    def __init__(self, in_channels, out_channels,pool_size = [1, 3, 5],in_channel_list=[],out_channel_list = [256, 128/2],cascade=False):
        super(CCM, self).__init__()
        self.cascade=cascade
        self.in_channel_list=in_channel_list
        self.out_channel_list = out_channel_list
        upsampe_scale = [2,4,8,16]
        GClist = []
        GCoutlist = []

        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(len(self.out_channel_list)):
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, self.out_channel_list[i], 3, 1, 1),
                                           nn.BatchNorm2d(self.out_channel_list[i]),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout2d(),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)


    def forward(self, x,y=None):
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule) - 1):
          global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        global_context.append(self.GCmodule[-1](x))
        global_context = torch.cat(global_context, dim=1)
        output = []
        for i in range(len(self.GCoutmodel)):
            out=self.GCoutmodel[i](global_context)
            if self.cascade is True and y is not None:
              out=out+y[i]     
            output.append(out)
        return output


#Balancing Attention Module

class BAM(nn.Module):
    def __init__(self,in_channels):
        super(BAM, self).__init__()
        
        self.boundary_conv=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//3,3,1,1),
            nn.BatchNorm2d(in_channels//3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.foregound_conv=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//3,3,1, 1),
            nn.BatchNorm2d(in_channels//3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.background_conv=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//3,3,1, 1),
            nn.BatchNorm2d(in_channels//3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.out_conv=nn.Sequential(
            nn.Conv2d((in_channels//3)*3, in_channels,3,1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.selayer=SELayer((in_channels//3)*3)

    def forward(self, x, pred):
        residual = x
        score = torch.sigmoid(pred)
        
        #boundary
        dist = torch.abs(score - 0.5)
        boundary_att = 1 - (dist / 0.5)
        boundary_x = x * boundary_att
        
        #foregound
        foregound_att= score
        foregound_att=torch.clip(foregound_att-boundary_att,0,1)
        foregound_x= x*foregound_att

        #background
        background_att=1-score
        background_att=torch.clip(background_att-boundary_att,0,1)
        background_x= x*background_att

        foregound_x= foregound_x 
        background_x= background_x 
        boundary_x= boundary_x  

        foregound_xx=self.foregound_conv(foregound_x)
        background_xx=self.background_conv(background_x)
        boundary_xx=self.boundary_conv(boundary_x)

        out=torch.cat([foregound_xx,background_xx,boundary_xx], dim=1) 
        out=self.selayer(out)
        out=self.out_conv(out)+residual
        return out