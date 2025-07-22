import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Attention.MSE import MultiScaleSEBlock
from model.Attention.SCAM import SCAM

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_scam=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
   
        self.use_scam = use_scam
        if use_scam:
            self.scam = SCAM(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        if self.use_scam:
            x = self.scam(x)  
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_rcssc=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_rcssc=use_rcssc)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, use_rcssc=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_rcssc=use_rcssc)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_rcssc=use_rcssc)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Lite1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Lite1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        
        self.inc = DoubleConv(n_channels, 32, use_scam=True)
        self.down1 = Down(32, 64, use_scam=True)
        self.down2 = Down(64, 128, use_scam=True)
        self.down3 = Down(128, 256, use_scam=False)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor, use_scam=False)  
        
        
        self.se1 = MultiScaleSEBlock(in_channels=256)
        self.se2 = MultiScaleSEBlock(in_channels=128)
        self.se3 = MultiScaleSEBlock(in_channels=64)
        
        
       
        self.up1 = Up(512, 256 // factor, bilinear, use_scam=False)  
        self.up2 = Up(256, 128 // factor, bilinear, use_scam=False)
        self.up3 = Up(128, 64, bilinear, use_scam=True)
        self.up4 = Up(64, 32, bilinear, use_scam=True)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  
        
        
        x = self.up1(x5, self.se1(x4))  
        x = self.up2(x, self.se2(x3))   
        x = self.up3(x, self.se3(x2))   
        x = self.up4(x, x1)             
        logits = self.outc(x)
        return logits

