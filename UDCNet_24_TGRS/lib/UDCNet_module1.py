import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class LE(nn.Module): # Local enhancement operation(LE)
    def __init__(self, in_channels):
        super(LE, self).__init__()

        self.Dconv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=3,dilation=3), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU(True)
        )
        self.Dconv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=5,dilation=5), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=7,dilation=7), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv9 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=9,dilation=9), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv12 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=12, dilation=12), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels * 3, 1), nn.BatchNorm2d(in_channels * 3),
            nn.Conv2d(in_channels * 3, in_channels * 2, 3, padding=1), nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

    def forward(self, F1):
       F1_3 = self.Dconv3(F1)
       F1_5 = self.Dconv5(F1+F1_3)
       F1_7 = self.Dconv7(F1+F1_5)
       F1_9 = self.Dconv9(F1+F1_7)
       F1_12 = self.Dconv12(F1+F1_9)
       out = self.out(torch.cat((F1_3,F1_5,F1_7,F1_9,F1_12),1)) + F1

       return out

    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):  #Cross-domian Feed-Forward Network
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        self.dwconv0 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.project_in  = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.LE = LE(dim)
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x    = self.project_in(x)
        x_f  = torch.fft.fft2(x.float())
        x_f1 = self.relu(self.norm(torch.abs(x_f)))
        x_f_gelu = F.gelu(x_f1)*x_f1
        x_f_1  = x_f_gelu
        x_f_2  = torch.fft.ifft2(torch.fft.fft2(x_f_1.float()))
        x_f    = self.relu(self.norm(torch.abs(x_f_2)))

        x_s1 = self.dwconv0(x)
        x_s    = F.gelu(x_s1)*x_s1
        x_s    = self.LE(x_s)
        out = self.project_out(torch.cat((x_f,x_s),1))

        return out

    def initialize(self):
        weight_init(self)


def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor

class Attention_F(nn.Module): # Frequency perception self-attention(FPSA)
    def __init__(self, dim, num_heads, bias):
        super(Attention_F, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.PConv_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.PConv_2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.PConv_3 = nn.Conv2d(dim, dim, kernel_size=1)
    def forward(self, x):
        b, c, h, w = x.shape

        q_f = torch.fft.fft2(self.PConv_1(x).float())
        k_f = torch.fft.fft2(self.PConv_2(x).float())
        v_f = torch.fft.fft2(self.PConv_3(x).float())

        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f = torch.nn.functional.normalize(q_f, dim=-1)
        k_f = torch.nn.functional.normalize(k_f, dim=-1)
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature
        attn_f = custom_complex_normalization(attn_f, dim=-1)
        out_fa = torch.abs(torch.fft.ifft2(attn_f @ v_f))
        out_f  = out_fa
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out   = self.project_out(out_f)
        return out

    def initialize(self):
        weight_init(self)


class Attention_S(nn.Module): # Spatial perception self-attention(SPSA)
    def __init__(self, dim, num_heads, bias):
        super(Attention_S, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.PConv1 = nn.Conv2d(dim,dim,kernel_size=1)
        self.PConv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.PConv3 = nn.Conv2d(dim, dim, kernel_size=1)


        self.DWConv3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.DWConv3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.DWConv3_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.DWConv5_1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.DWConv5_2 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.DWConv5_3 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)

        self.DWConv7_1 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.DWConv7_2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.DWConv7_3 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q_s = self.DWConv3_1(self.PConv1(x))+self.DWConv5_1(self.PConv1(x))+self.DWConv7_1(self.PConv1(x))
        k_s = self.DWConv3_2(self.PConv2(x))+self.DWConv5_2(self.PConv2(x))+self.DWConv7_2(self.PConv2(x))
        v_s = self.DWConv3_3(self.PConv3(x))+self.DWConv5_3(self.PConv3(x))+self.DWConv7_3(self.PConv3(x))

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = attn_s.softmax(dim=-1)
        out_s = (attn_s @ v_s)
        out_s = rearrange(out_s, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out   = self.project_out(out_s)
        return out

    def initialize(self):
        weight_init(self)

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

class HIM(nn.Module): # Adaptive fusion strategy (AFS)
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

class HIM3(nn.Module): # Adaptive fusion strategy (AFS)
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

class HIM4(nn.Module): # Adaptive fusion strategy (AFS)
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

class Module_1(nn.Module):
    def __init__(self, dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,LayerNorm_type='WithBias'):
        super(Module_1, self).__init__()
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_S = Attention_S(dim, num_heads, bias)
        self.attn_F = Attention_F(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.HIM   = HIM(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.HIM(self.attn_F(self.norm1(x)),self.attn_S(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)


class Module1_res(nn.Module): # Frequency-spatial domain transformer (FSDT) block
    def __init__(self, in_channel, out_channel):
        super(Module1_res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True))
        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 2, 1), nn.BatchNorm2d(out_channel * 2),
            nn.Conv2d(out_channel * 2, out_channel*2, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel*2),
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel),nn.ReLU(True),)

        self.reduce  = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 1),nn.BatchNorm2d(out_channel), nn.ReLU(True))
        self.module1 = Module_1(dim=out_channel)

    def forward(self, x):
        x0    = self.conv1(x)
        x_FT  = self.module1(x0)
        x_res = self.res(x)
        x     = self.reduce(torch.cat((x_res,x_FT),1))+x0
        return x

