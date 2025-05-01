import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.module_fusion_RSSOD import HIM, HIM3, HIM4

class DSE_module(nn.Module): # Dense semantic excavation module(DSE_module)
    def __init__(self, inchannels, depth=128):

        super(DSE_module, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inchannels, depth, kernel_size=1, stride=1),nn.BatchNorm2d(depth),nn.ReLU(True)
        )
        self.branch_main_mp = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(inchannels, depth, kernel_size=1, stride=1), nn.BatchNorm2d(depth), nn.ReLU(True)
        )

        self.branch0 = nn.Sequential(nn.Conv2d(inchannels, depth, kernel_size=1, stride=1),nn.BatchNorm2d(depth),nn.ReLU(True))
        self.branch1 = nn.Sequential(nn.Conv2d(inchannels+depth, depth, kernel_size=3, stride=1, padding=3,dilation=3), nn.BatchNorm2d(depth), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(inchannels+depth*2, depth, kernel_size=3, stride=1, padding=6,dilation=6),nn.BatchNorm2d(depth),nn.ReLU(True))
        self.branch3 = nn.Sequential(nn.Conv2d(inchannels+depth*3, depth, kernel_size=3, stride=1, padding=12,dilation=12),nn.BatchNorm2d(depth),nn.ReLU(True))
        self.branch4 = nn.Sequential(nn.Conv2d(inchannels+depth*4, depth, kernel_size=3, stride=1, padding=18, dilation=18),nn.BatchNorm2d(depth),nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Conv2d(depth * 6, depth*3, kernel_size=3, padding=1),nn.BatchNorm2d(depth*3),
            nn.Conv2d(depth * 3, depth*2, kernel_size=3, padding=1), nn.BatchNorm2d(depth*2),
            nn.Conv2d(depth * 2, depth, kernel_size=3, padding=1), nn.BatchNorm2d(depth), nn.ReLU(True)
        )
        self.res = nn.Sequential(
            nn.Conv2d(inchannels, depth*2, kernel_size=1, stride=1), nn.BatchNorm2d(depth*2),
            nn.Conv2d(depth*2, depth*2, kernel_size=3, stride=1,padding=1), nn.BatchNorm2d(depth*2),
            nn.Conv2d(depth*2, depth, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(depth), nn.ReLU(True)
                                 )
        self.out = nn.Sequential(
            nn.Conv2d(depth, depth//2, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)+self.branch_main_mp(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        branch0 = self.branch0(x)
        branch1 = self.branch1(torch.cat((x,branch0),1))
        branch2 = self.branch2(torch.cat((x,branch0,branch1),1))
        branch3 = self.branch3(torch.cat((x,branch0,branch1,branch2),1))
        branch4 = self.branch4(torch.cat((x,branch0,branch1,branch2,branch3),1))
        out = torch.cat([branch_main, branch0, branch1, branch2, branch3, branch4], 1)
        out = self.head(out)+self.res(x)
        out = self.out(out)
        return out

class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa(x)
        y = x * out
        return y

class Edge_EH(nn.Module):
    def __init__(self, channels):
        super(Edge_EH, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=1), nn.BatchNorm2d(channels*2),
            nn.Conv2d(channels*2, channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1), nn.BatchNorm2d(channels),nn.ReLU(True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1), nn.BatchNorm2d(channels * 2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=1), nn.BatchNorm2d(channels*2),
            nn.Conv2d(channels*2, channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(channels),nn.ReLU(True))
        self.SA = SA(channels)

        self.out = nn.Sequential(
            nn.Conv2d(channels*2,channels,kernel_size=1),nn.BatchNorm2d(channels),nn.ReLU(True)
        )

    def forward(self, x):
        x1 = self.SA(self.conv0(x))
        x2 = self.conv(self.conv1(x))
        y  = self.out(torch.cat((x1,x2),1))
        return y

class Module_2_1(nn.Module): # Dual_branch joint Optimization (DJO) Decoder
    def __init__(self, in_channels, mid_channels):
        super(Module_2_1, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),nn.BatchNorm2d(mid_channels//2),nn.ReLU(True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=1), nn.BatchNorm2d(in_channels*2),
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels*2),
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),nn.ReLU(True),
        )

        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels//2, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels//2),
            nn.Conv2d(mid_channels//2, mid_channels // 4, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels//4),nn.ReLU(True),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=3, padding=1)
        )
        self.edge = Edge_EH(in_channels)
        self.HIM  = HIM(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)

    def forward(self, X, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)
        FI  = X

        yt  = self.HIM(FI,prior_cam.expand(-1, X.size()[1], -1, -1))

        out_edge = self.edge(yt)
        edge    = self.out_edge(out_edge)
        edge    = self.edge_enhance(edge)

        yt_t = self.conv3(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(torch.fft.fft2(yt.float())))))
        yt_out = torch.add(yt_t,yt_f)

        r_prior_cam_f = torch.abs(torch.fft.fft2(prior_cam))
        r_prior_cam_f = -1 * (torch.sigmoid(r_prior_cam_f)) + 1
        r_prior_cam_t = -1 * (torch.sigmoid(prior_cam)) + 1
        r_prior_cam = r_prior_cam_t+r_prior_cam_f

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([out_edge, y_r, yt_out], dim=1)

        y = self.out(cat2)
        y = y + prior_cam
        return y,edge

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class Module_2_2(nn.Module): # Dual_branch joint Optimization (DJO) Decoder
    def __init__(self, in_channels, mid_channels):
        super(Module_2_2, self).__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels // 2),nn.ReLU(True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )
        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels // 2, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels // 2),
            nn.Conv2d(mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1),nn.BatchNorm2d(mid_channels//4), nn.ReLU(True),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=3, padding=1)
        )
        self.edge = Edge_EH(in_channels)
        self.HIM  = HIM3(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)

    def forward(self, X, x1, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt  = self.HIM(FI,prior_cam.expand(-1, X.size()[1], -1, -1),x1_prior_cam.expand(-1, X.size()[1], -1, -1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(torch.fft.fft2(yt.float())))))
        yt_out = torch.add(yt_t, yt_f)

        r_prior_cam_f = torch.abs(torch.fft.fft2(prior_cam))
        r_prior_cam_f = -1 * (torch.sigmoid(r_prior_cam_f)) + 1
        r_prior_cam_t = -1 * (torch.sigmoid(prior_cam)) + 1
        r_prior_cam = r_prior_cam_t+r_prior_cam_f

        r1_prior_cam_f = torch.abs(torch.fft.fft2(x1_prior_cam))
        r1_prior_cam_f = -1 * (torch.sigmoid(r1_prior_cam_f)) + 1
        r1_prior_cam_t = -1 * (torch.sigmoid(x1_prior_cam)) + 1
        r1_prior_cam = r1_prior_cam_t+r1_prior_cam_f

        r_prior_cam = r_prior_cam + r1_prior_cam

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out,out_edge], dim=1)  #

        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam
        return y,edge
    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class Module_2_3(nn.Module): # Dual_branch joint Optimization (DJO) Decoder
    def __init__(self, in_channels, mid_channels):
        super(Module_2_3, self).__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels // 2),nn.ReLU(True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )
        self.out_edge = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels // 2, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels // 2),
            nn.Conv2d(mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1),nn.BatchNorm2d(mid_channels//4), nn.ReLU(True),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=3, padding=1)
        )
        self.edge = Edge_EH(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
        self.HIM  = HIM4(in_channels)
    def forward(self, X, x1, x2, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X
        yt = self.HIM(FI,prior_cam.expand(-1, X.size()[1], -1, -1),x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1))

        out_edge = self.edge(yt)
        edge = self.out_edge(out_edge)
        edge = self.edge_enhance(edge)

        yt_t = self.conv3(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(torch.fft.fft2(yt.float())))))
        yt_out = torch.add(yt_t, yt_f)

        r_prior_cam_f = torch.abs(torch.fft.fft2(prior_cam))
        r_prior_cam_f = -1 * (torch.sigmoid(r_prior_cam_f)) + 1
        r_prior_cam_t = -1 * (torch.sigmoid(prior_cam)) + 1
        r_prior_cam = r_prior_cam_t+r_prior_cam_f

        r1_prior_cam_f = torch.abs(torch.fft.fft2(x1_prior_cam))
        r1_prior_cam_f = -1 * (torch.sigmoid(r1_prior_cam_f)) + 1
        r1_prior_cam_t = -1 * (torch.sigmoid(x1_prior_cam)) + 1
        r1_prior_cam1 = r1_prior_cam_t+r1_prior_cam_f

        r2_prior_cam_f = torch.abs(torch.fft.fft2(x2_prior_cam))
        r2_prior_cam_f = -1 * (torch.sigmoid(r2_prior_cam_f)) + 1
        r2_prior_cam_t = -1 * (torch.sigmoid(x2_prior_cam)) + 1
        r1_prior_cam2 = r2_prior_cam_t + r2_prior_cam_f

        r_prior_cam = r_prior_cam + r1_prior_cam1+r1_prior_cam2

        y_r = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        cat2 = torch.cat([y_r, yt_out,out_edge], dim=1)
        y = self.out(cat2)
        y = y + prior_cam + x1_prior_cam + x2_prior_cam
        return y,edge

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out
















