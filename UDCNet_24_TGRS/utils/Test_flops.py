import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
#from lib.Network2_Swin import Network
from Src.FSPNet_model import Model
import torch
import torchvision
from thop import profile

# Model
print('==> Building model..')
model = Model(img_size=384)

dummy_input = torch.randn(1, 3, 384, 384)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))



