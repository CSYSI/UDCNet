import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.UDCNet_module1 import Module1_res
from lib.UDCNet_module2 import Module_2_1, Module_2_2, Module_2_3, DSE_module






class Network(nn.Module):
    def __init__(self, channels=128):
        super(Network, self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

        self.dePixelShuffle = torch.nn.PixelShuffle(2)

        self.up = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1),nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),nn.BatchNorm2d(channels),nn.ReLU(True)
        )

        self.FSDT5 = Module1_res(2048, channels)
        self.FSDT4 = Module1_res(1024+channels, channels)
        self.FSDT3 = Module1_res(512+channels, channels)
        self.FSDT2 = Module1_res(256+channels, channels)

        self.DSE = DSE_module(2048+channels)

        self.DJO_1 = Module_2_1(channels, channels)
        self.DJO_2 = Module_2_2(channels, channels)
        self.DJO_3 = Module_2_3(channels,channels)

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        x4_h1 = x4

        x4   = self.FSDT5(x4)
        x4_h2 = x4
        x4_up = self.up(self.dePixelShuffle(x4))

        p1 = self.DSE(torch.cat((x4_h1,x4_h2),1))
        x5_4 = p1

        x3   = self.FSDT4(torch.cat((x3,x4_up),1))
        x3_up = self.up(self.dePixelShuffle(x3))

        x2   = self.FSDT3(torch.cat((x2,x3_up),1))
        x2_up = self.up(self.dePixelShuffle(x2))

        x1   = self.FSDT2(torch.cat((x1,x2_up),1))

        x4,e4 = self.DJO_1(x4,x5_4)
        x3,e3 = self.DJO_1(x3,x4)
        x2,e2 = self.DJO_2(x2,x3,x4)
        x1,e1 = self.DJO_3(x1,x2,x3,x4)

        p0 = F.interpolate(p1, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)

        e4 = F.interpolate(e4, size=image.size()[2:], mode='bilinear', align_corners=True)
        e3 = F.interpolate(e3, size=image.size()[2:], mode='bilinear', align_corners=True)
        e2 = F.interpolate(e2, size=image.size()[2:], mode='bilinear', align_corners=True)
        e1 = F.interpolate(e1, size=image.size()[2:], mode='bilinear', align_corners=True)

        return p0, f4, f3, f2, f1, e4, e3, e2, e1

