import torch

# 假设特征张量为 input_feature，形状为 (3, 128, 128)
input_feature = torch.randn(3, 128, 128)


x = torch.randn(10,49,128)
B,N,C = x.shape
W = 7
H = 7
x_l = x.permute(0,2,1).reshape(B,C,H,W)
Conv3 = torch.nn.Conv2d(128,128,)
x_l_1 = x_l.flatten(2).transpose(1, 2)




# 应用FFT进行频域变换
#fft_feature = torch.fft.rfft2(input_feature, dim=(-2, -1))
#fft_2 = torch.fft.irfft2(fft_feature, dim=(-2, -1))

# 输出为实数特征，形状为 (3, 128, 128)
print(x_l.shape)
print(x_l_1.shape)