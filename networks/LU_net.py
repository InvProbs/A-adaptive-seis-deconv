import torch.nn as nn
import torch


class single_layer_conv2D_y(nn.Module):
    def __init__(self, args):
        super(single_layer_conv2D_y, self).__init__()
        ks = args.kernel_size
        pad = args.padding
        model = nn.Sequential(nn.Conv2d(2, 64, kernel_size=ks, padding=pad),
                              nn.GroupNorm(4, 64),
                              nn.GELU(),
                              nn.Conv2d(64, 64, kernel_size=ks, padding=pad),
                              nn.GroupNorm(4, 64),
                              nn.GELU(),
                              nn.Conv2d(64, 64, kernel_size=ks, padding=pad),
                              nn.GroupNorm(4, 64),
                              nn.GELU(),
                              nn.Conv2d(64, 1, kernel_size=ks, padding=pad),
                              nn.Conv2d(1, 1, kernel_size=1),
                              )
        self.data_layer = nn.Sequential(*model)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            with torch.no_grad():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.001)
                    m.weight /= 10
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, y, xk):
        inp = self.data_layer(torch.cat((y, xk), dim=1))
        return inp



class inverse_block_prox2D_y(nn.Module):
    """ compute x[k+1] = forward_module(x[k]) - eta * R(x[k])"""

    def __init__(self, linear_op, DnCNN, args):
        super(inverse_block_prox2D_y, self).__init__()
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self.linear_op = linear_op
        self.R = DnCNN
        self.sigmoid = nn.Sigmoid()

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def set_initial_point(self, y):
        self.initial_point = self._linear_adjoint(y)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z) - self._linear_adjoint(y)

    def forward_module(self, x, y):
        return x - 0.1 * self.sigmoid(self.eta) * self.get_gradient(x, y)
        # return x - torch.exp(self.eta) * self.get_gradient(x, y)

    def forward(self, xk, y, train):  # [bs, 1, 200, 1], [bs, 1, 200, 1]
        dk = self.forward_module(xk, y)  # [64, 1, 200, 1]
        return self.R(y, dk)  # + forward_res



""" 
(2D) ADDING y INJECTION TO PROXIMAL OPERATOR 
"""


def ConvBlock2D(in_chans, out_chans, stride, padding, drop_prob):
    conv = nn.Sequential(
        nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(out_chans),
        nn.ReLU(),
    )
    return conv


def TransposeConvBlock2D(in_chans, out_chans, stride, padding, output_padding):
    trans = nn.Sequential(
        nn.ConvTranspose2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=padding,
                           output_padding=output_padding),
        nn.BatchNorm2d(out_chans),
    )
    return trans

""" UNet """
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.cb1 = ConvBlock2D(in_chans=1, out_chans=8, stride=1, padding=1, drop_prob=0.1)
        self.cb2 = ConvBlock2D(in_chans=8, out_chans=16, stride=2, padding=1, drop_prob=0.1)
        self.cb3 = ConvBlock2D(in_chans=16, out_chans=16, stride=1, padding=1, drop_prob=0.1)
        self.cb4 = ConvBlock2D(in_chans=16, out_chans=32, stride=2, padding=1, drop_prob=0.1)
        self.cb5 = ConvBlock2D(in_chans=32, out_chans=32, stride=1, padding=1, drop_prob=0.1)
        self.cb6 = ConvBlock2D(in_chans=32, out_chans=64, stride=2, padding=1, drop_prob=0.1)
        self.cb7 = ConvBlock2D(in_chans=64, out_chans=64, stride=1, padding=1, drop_prob=0.1)
        self.cb8 = ConvBlock2D(in_chans=64, out_chans=128, stride=2, padding=1, drop_prob=0.1)
        self.cb9 = ConvBlock2D(in_chans=128, out_chans=128, stride=1, padding=1, drop_prob=0.1)
        self.cb10 = ConvBlock2D(in_chans=128, out_chans=256, stride=2, padding=1, drop_prob=0.1)
        self.cb11 = ConvBlock2D(in_chans=256, out_chans=256, stride=1, padding=1, drop_prob=0.1)

        self.t1 = TransposeConvBlock2D(in_chans=256, out_chans=256, stride=1, padding=1, output_padding=0)
        self.t2 = TransposeConvBlock2D(in_chans=256, out_chans=128, stride=2, padding=1, output_padding=1)
        # self.t3 = ConvBlock2D(in_chans=128, out_chans=128, stride=1, padding=1, drop_prob=0.1)
        self.t3 = TransposeConvBlock2D(in_chans=128, out_chans=64, stride=2, padding=1, output_padding=1)
        self.t4 = ConvBlock2D(in_chans=64, out_chans=64, stride=1, padding=1, drop_prob=0.1)
        self.t5 = TransposeConvBlock2D(in_chans=64, out_chans=32, stride=2, padding=1, output_padding=1)
        self.t6 = ConvBlock2D(in_chans=32, out_chans=32, stride=1, padding=1, drop_prob=0.1)
        self.t7 = TransposeConvBlock2D(in_chans=32, out_chans=16, stride=2, padding=1, output_padding=1)
        self.t8 = ConvBlock2D(in_chans=16, out_chans=16, stride=1, padding=1, drop_prob=0.1)
        self.t9 = TransposeConvBlock2D(in_chans=16, out_chans=8, stride=2, padding=1, output_padding=1)
        self.out_layer = nn.Conv2d(8, 1, (1, 1), stride=(1, 1))

        self.relu = nn.ReLU()

    def forward(self, x0):
        x1 = self.cb1(x0)  # [8, 8, 352, 352]
        x2 = self.cb2(x1)  # [8, 16, 176, 176]
        x3 = self.cb3(x2)  # [8, 16, 176, 176]
        x4 = self.cb4(x3)  # [8, 32, 88, 88]
        x5 = self.cb5(x4)  # [8, 32, 88, 88]
        x6 = self.cb6(x5)  # [8, 64, 44, 44]
        x7 = self.cb7(x6)  # [8, 64, 44, 44]
        x8 = self.cb8(x7)  # [8, 128, 22, 22]
        x9 = self.cb9(x8)  # [8, 128, 22, 22]
        x10 = self.cb10(x9)  # [8, 256, 11, 11]
        x11 = self.cb11(x10) # [8, 256, 11, 11]

        x12 = self.relu(self.t1(x11))  # [8, 256, 11, 11]
        x13 = self.relu(self.t2(x12))  # [8, 128, 22, 22]
        x14 = self.t3(x9 + x13)  # [8, 64, 44, 44]
        x15 = self.relu(self.t4(x7 + x14))  # [8, 64, 44, 44]
        x16 = self.t5(x7 + x15)  # [8, 32, 88, 88]
        x17 = self.relu(self.t6(x5 + x16))  # [8, 32, 88, 88]
        x18 = self.t7(x5 + x17)  # [8, 16, 176, 176]
        x19 = self.relu(self.t8(x3 + x18))  # [8, 16, 176, 176]
        x20 = self.t8(x3 + x19) # [8, 16, 176, 176]

        x21 = self.relu(self.t9(x2 + x20))
        x22 = self.out_layer(x21)  #
        return x22
