import torch
import torch.nn as nn
import math

class ResBLockDB(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBLockDB, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out

class ReconstructModule(nn.Module):
    def __init__(self):
        super(ReconstructModule, self).__init__()
        self.resBlock = self._makelayers(64, 64, 8)
        self.conv1 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle1 = nn.PixelShuffle(2)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle2 = nn.PixelShuffle(2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, 1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(64, 3, (3, 3), 1, 1)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        res1 = self.resBlock(x)
        con1 = self.conv1(res1)
        pixelshuffle1 = self.relu1(self.pixelShuffle1(con1))
        con2 = self.conv2(pixelshuffle1)
        pixelshuffle2 = self.relu2(self.pixelShuffle2(con2))
        con3 = self.relu3(self.conv3(pixelshuffle2))
        sr_deblur = self.conv4(con3)
        return sr_deblur