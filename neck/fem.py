import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..utils import Conv_BN_ReLU
from ..utils import Dilated_Conv
from ..utils import ACConv2d


class FEM_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FEM_v2, self).__init__()

        # Top layer
        self.toplayer_ = Conv_BN_ReLU(2048, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth2_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth3_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1_ = Conv_BN_ReLU(1024, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer2_ = Conv_BN_ReLU(512, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer3_ = Conv_BN_ReLU(256, 256, kernel_size=1, stride=1, padding=0)

        # Dilated_conv
        self.dilated1 = Dilated_Conv(256, 256, kernel_size=3, stride=1, padding=2)

        self.dilated2 = Dilated_Conv(256, 256, kernel_size=3, stride=1, padding=2)

        self.dilated3 = Dilated_Conv(256, 256, kernel_size=3, stride=1, padding=2)

        # AC_conv
        self.acconv = ACConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, f2, f3, f4, f5):
        p5 = self.toplayer_(f5)

        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4)
        pp4 = f4 + p4
        PPP4 = self.acconv(pp4)
        p4 = self.dilated1(p4)
        p4 = pp4 + p4
        p4 = self.smooth1_(p4)

        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3)
        pp3 = f3 + p3
        PPP3 = self.acconv(pp3)
        p3 = self.dilated1(p3)
        p3 = pp3 + p3
        p3 = self.smooth2_(p3)

        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2)
        pp2 = f2 + p2
        PPP2 = self.acconv(pp2)
        p2 = self.dilated1(p2)
        p2 = pp2 + p2
        p2 = self.smooth3_(p2)

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        return p2, p3, p4, p5
