import torch
import torch.nn as nn

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

class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)+self.mp(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class AM(nn.Module):
    def __init__(self, channels):
        super(AM, self).__init__()
        self.CA = CA()
        self.SA = SA(channels)

    def forward(self, x):
        x_res = x
        x = self.CA(x)
        x = self.SA(x)
        return x+x_res

class HIM(nn.Module):
    def __init__(self, channels):
        super(HIM, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(channels * 4, channels*2, kernel_size=1),nn.BatchNorm2d(channels*2),
            nn.Conv2d(channels * 2, channels*2, kernel_size=3,padding=1), nn.BatchNorm2d(channels*2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True)
        )
        self.JA  = AM(channels)

    def forward(self, x_1,x_2):
        x_m = x_1*x_2
        x_a = x_1+x_2
        x_c = self.out(torch.cat((x_m,x_a,x_1,x_2),1))
        x = self.JA(x_c)
        return x

class HIM3(nn.Module):
    def __init__(self, channels):
        super(HIM3, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(channels * 5, channels * 3, kernel_size=1), nn.BatchNorm2d(channels * 3),
            nn.Conv2d(channels * 3, channels * 2, kernel_size=3, padding=1), nn.BatchNorm2d(channels * 2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True)
        )
        self.JA  = AM(channels)

    def forward(self, x_1,x_2,x_3):
        x_m = x_1*x_2*x_3
        x_a = x_1+x_2+x_3
        x_c = self.out(torch.cat((x_m,x_a,x_1,x_2,x_3),1))
        x = self.JA(x_c)
        return x

class HIM4(nn.Module):
    def __init__(self, channels):
        super(HIM4, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(channels * 6, channels * 3, kernel_size=1), nn.BatchNorm2d(channels*3),
            nn.Conv2d(channels * 3, channels * 2, kernel_size=3, padding=1), nn.BatchNorm2d(channels*2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True)
        )
        self.JA  = AM(channels)

    def forward(self, x_1,x_2,x_3,x_4):
        x_m = x_1*x_2*x_3*x_4
        x_a = x_1+x_2+x_3+x_4
        x_c = self.out(torch.cat((x_m,x_a,x_1,x_2,x_3,x_4),1))
        x = self.JA(x_c)
        return x






















