import torch
import torch.nn as nn

from torch.nn import functional as F

class AMS(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()

        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        auto_padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=auto_padding, groups=dim, dilation=dilation)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv(x)

        attna = self.conv0(attn)

        attn_0 = self.conv0_1(attna)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attna)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attna)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attna + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class Attention(nn.Module):
    def __init__(self, d_model, dilation=1):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv2d(d_model, d_model, 1)
        self.act = nn.GELU()
        self.ams = AMS(d_model, dilation=dilation)
        self.conv2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.ams(x)
        x = self.conv2(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = d_model * 4
        self.conv1 = nn.Conv2d(d_model, hidden_dim, 1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(hidden_dim, d_model, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x


class AMSE(nn.Module):
    def __init__(self, dim, dilation=1):
        super(AMSE, self).__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(dim)
        self.ffn = FFN(dim)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.bn1(x)
        x = self.attn(x)
        x = self.bn2(x)
        x = self.ffn(x)
        x = x + shortcut
        x = self.act(x)
        return x


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, layer_num=1, dilation=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(AMSE(out_channels, dilation=dilation))
        super(Down, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            *layers
        )


class MSAG(nn.Module):
    """
    Multi-scale attention gate
    """

    def __init__(self, channel):
        super(MSAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 3, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        _x = self.relu(torch.cat((x1, x2, x3), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x



class BlockTypeA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale = True):
        super(BlockTypeA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True)
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        # a = self.conv2(a)
        if self.upscale:
            b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)


class BlockTypeB(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x

class BlockTypeC(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class MobileV2_MLSD_Large(nn.Module):
    def __init__(self,cfg=None, base_c=32):
        super(MobileV2_MLSD_Large, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(4, base_c, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(base_c),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            AMSE(base_c),
        )

        self.down1 = Down(base_c, base_c * 2, dilation=1)
        self.down2 = Down(base_c * 2, base_c * 4, dilation=1)
        self.down3 = Down(base_c * 4, base_c * 8, dilation=3)
        self.down4 = Down(base_c * 8, base_c * 16, dilation=5)

        # Skip-connection
        self.msag4 = MSAG(256)
        self.msag3 = MSAG(128)
        self.msag2 = MSAG(64)
        self.msag1 = MSAG(32)

        ## A, B
        self.block15_a = BlockTypeA(in_c1=256, in_c2=512,
                                  out_c1=256, out_c2=256, )
        self.block16_b = BlockTypeB(512, 256)

        ## A, B
        self.block17_a = BlockTypeA(in_c1=128, in_c2=256,
                                  out_c1=128, out_c2=128)
        self.block18_b = BlockTypeB(256, 128)

        ## A, B
        self.block19_a = BlockTypeA(in_c1=64, in_c2=128,
                                  out_c1=64, out_c2=64)
        self.block20_b = BlockTypeB(128, 64)

        ## A, B, C
        self.block21_a = BlockTypeA(in_c1=32, in_c2=64,
                                  out_c1=32, out_c2=32)
        self.block22_b = BlockTypeB(64, 32)

        self.block23_c = BlockTypeC(32, 16)

        self.block24_c = BlockTypeC(32, 16)
        self.block25_c = BlockTypeC(16, 1)

    def forward(self, x):
        c1 = self.in_conv(x)  # (1,32,256,256)
        c2 = self.down1(c1)  # (1,64,128,128)
        c3 = self.down2(c2)  # (1,128,64,64)
        c4 = self.down3(c3)  # (1,256,32,32)
        c5 = self.down4(c4)  # (1,512,16,16)

        c4 = self.msag4(c4)
        c3 = self.msag3(c3)
        c2 = self.msag2(c2)
        c1 = self.msag1(c1)

        x = self.block15_a(c4, c5)  # (1,512,32,32) #c5上采样，通道减半；c4不变；cat
        x = self.block16_b(x)  # (1,256,32,32)#通道减半

        x = self.block17_a(c3, x)  # (1,256,64,64)
        x = self.block18_b(x)  # (1,128,64,64)

        x = self.block19_a(c2, x)  # (1,128,128,128)
        fea = self.block20_b(x)  # (1,64,128,128)

        x = self.block21_a(c1, fea)  # (1,64,256,256)
        x1 = self.block22_b(x)  # (1,32,256,256)

        x = self.block23_c(x1)  # (1,16,256,256)
        # x = x[:, 7:, :, :]

        seg_x = self.block24_c(x1)
        seg_x = self.block25_c(seg_x)

        return x, torch.sigmoid(seg_x)




if __name__ == '__main__':
    model = MobileV2_MLSD_Large()
    input = torch.randn(1, 4, 512, 512)
    out = model(input)
    print(out[0].shape)


# import torch
# import torch.nn as nn
#
# from torch.nn import functional as F
#
# class AMS(nn.Module):
#     def __init__(self, dim, kernel_size=3, dilation=1):
#         super().__init__()
#
#         self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
#         auto_padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
#         self.conv0 = nn.Conv2d(dim, dim, 3, padding=auto_padding, groups=dim, dilation=dilation)
#
#         self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
#         self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
#
#         self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
#
#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv(x)
#
#         attna = self.conv0(attn)
#
#         attn_0 = self.conv0_1(attna)
#         attn_0 = self.conv0_2(attn_0)
#
#         attn_1 = self.conv1_1(attna)
#         attn_1 = self.conv1_2(attn_1)
#
#         attn_2 = self.conv2_1(attna)
#         attn_2 = self.conv2_2(attn_2)
#
#         attn = attn + attna + attn_0 + attn_1 + attn_2
#
#         attn = self.conv3(attn)
#
#         return attn * u
#
#
# class Attention(nn.Module):
#     def __init__(self, d_model, dilation=1):
#         super().__init__()
#         self.d_model = d_model
#         self.conv1 = nn.Conv2d(d_model, d_model, 1)
#         self.act = nn.GELU()
#         self.ams = AMS(d_model, dilation=dilation)
#         self.conv2 = nn.Conv2d(d_model, d_model, 1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.act(x)
#         x = self.ams(x)
#         x = self.conv2(x)
#         return x
#
#
# class FFN(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         hidden_dim = d_model * 4
#         self.conv1 = nn.Conv2d(d_model, hidden_dim, 1)
#         self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
#         self.act = nn.GELU()
#         self.conv3 = nn.Conv2d(hidden_dim, d_model, 1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.act(x)
#         x = self.conv3(x)
#         return x
#
#
# class AMSE(nn.Module):
#     def __init__(self, dim, dilation=1):
#         super(AMSE, self).__init__()
#         self.bn1 = nn.BatchNorm2d(dim)
#         self.attn = Attention(dim, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(dim)
#         self.ffn = FFN(dim)
#         self.act = nn.GELU()
#
#     def forward(self, x):
#         shortcut = x
#         x = self.bn1(x)
#         x = self.attn(x)
#         x = self.bn2(x)
#         x = self.ffn(x)
#         x = x + shortcut
#         x = self.act(x)
#         return x
#
#
# class Down(nn.Sequential):
#     def __init__(self, in_channels, out_channels, layer_num=1, dilation=1):
#         layers = nn.ModuleList()
#         for i in range(layer_num):
#             layers.append(AMSE(out_channels, dilation=dilation))
#         super(Down, self).__init__(
#             nn.BatchNorm2d(in_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
#             *layers
#         )
#
# # class Down(nn.Sequential):
# #     def __init__(self, in_channels, out_channels, layer_num=1, dilation=1):
# #         layers = nn.ModuleList()
# #         for i in range(layer_num):
# #             layers.append(AMSE(out_channels, dilation=dilation))
# #         super(Down, self).__init__(
# #             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
# #             nn.BatchNorm2d(out_channels),
# #             nn.GELU(),
# #             *layers,
# #             nn.MaxPool2d(2,2)
# #         )
#
# class MSAG(nn.Module):
#     """
#     Multi-scale attention gate
#     """
#
#     def __init__(self, channel):
#         super(MSAG, self).__init__()
#         self.channel = channel
#         self.pointwiseConv = nn.Sequential(
#             nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
#             nn.BatchNorm2d(self.channel),
#         )
#         self.ordinaryConv = nn.Sequential(
#             nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
#             nn.BatchNorm2d(self.channel),
#         )
#         self.dilationConv = nn.Sequential(
#             nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
#             nn.BatchNorm2d(self.channel),
#         )
#         self.voteConv = nn.Sequential(
#             nn.Conv2d(self.channel * 3, self.channel, kernel_size=(1, 1)),
#             nn.BatchNorm2d(self.channel),
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x1 = self.pointwiseConv(x)
#         x2 = self.ordinaryConv(x)
#         x3 = self.dilationConv(x)
#         _x = self.relu(torch.cat((x1, x2, x3), dim=1))
#         _x = self.voteConv(_x)
#         x = x + x * _x
#         return x
#
#
#
# class BlockTypeA(nn.Module):
#     def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale = True):
#         super(BlockTypeA, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_c2, out_c2, kernel_size=1),
#             nn.BatchNorm2d(out_c2),
#             nn.ReLU(inplace=True)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_c1, out_c1, kernel_size=1),
#             nn.BatchNorm2d(out_c1),
#             nn.ReLU(inplace=True)
#         )
#         self.upscale = upscale
#
#     def forward(self, a, b):
#         b = self.conv1(b)
#         # a = self.conv2(a)
#         if self.upscale:
#             b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
#         return torch.cat((a, b), dim=1)
#
#
# class BlockTypeB(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(BlockTypeB, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_c),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         x = self.conv1(x) + x
#         x = self.conv2(x)
#         return x
#
# class BlockTypeC(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(BlockTypeC, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_c, in_c,  kernel_size=3, padding=5, dilation=5),
#             nn.BatchNorm2d(in_c),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_c),
#             nn.ReLU()
#         )
#         self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         return x
#
#
# class MobileV2_MLSD_Large(nn.Module):
#     def __init__(self,cfg=None, base_c=32):
#         super(MobileV2_MLSD_Large, self).__init__()
#
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(4, base_c, kernel_size=3, padding=1, padding_mode='reflect'),
#             nn.BatchNorm2d(base_c),
#             nn.GELU(),
#             AMSE(base_c),
#             nn.MaxPool2d(2, 2),
#         )
#
#         self.down1 = Down(base_c, base_c * 2, dilation=1)
#         self.down2 = Down(base_c * 2, base_c * 4, dilation=1)
#         self.down3 = Down(base_c * 4, base_c * 8, dilation=3)
#         self.down4 = Down(base_c * 8, base_c * 16, dilation=5)
#
#         # Skip-connection
#         self.msag4 = MSAG(256)
#         self.msag3 = MSAG(128)
#         self.msag2 = MSAG(64)
#         self.msag1 = MSAG(32)
#
#         ## A, B
#         self.block15_a = BlockTypeA(in_c1=256, in_c2=512,
#                                   out_c1=256, out_c2=256, )
#         self.block16_b = BlockTypeB(512, 256)
#
#         ## A, B
#         self.block17_a = BlockTypeA(in_c1=128, in_c2=256,
#                                   out_c1=128, out_c2=128)
#         self.block18_b = BlockTypeB(256, 128)
#
#         ## A, B
#         self.block19_a = BlockTypeA(in_c1=64, in_c2=128,
#                                   out_c1=64, out_c2=64)
#         self.block20_b = BlockTypeB(128, 64)
#
#         ## A, B, C
#         self.block21_a = BlockTypeA(in_c1=32, in_c2=64,
#                                   out_c1=32, out_c2=32)
#         self.block22_b = BlockTypeB(64, 32)
#
#         self.block23_c = BlockTypeC(32, 16)
#
#         self.block24_c = BlockTypeC(32, 16)
#         self.block25_c = BlockTypeC(16, 1)
#
#     def forward(self, x):
#         c1 = self.in_conv(x)  # (1,32,256,256)
#         c2 = self.down1(c1)  # (1,64,128,128)
#         c3 = self.down2(c2)  # (1,128,64,64)
#         c4 = self.down3(c3)  # (1,256,32,32)
#         c5 = self.down4(c4)  # (1,512,16,16)
#
#         c4 = self.msag4(c4)
#         c3 = self.msag3(c3)
#         c2 = self.msag2(c2)
#         c1 = self.msag1(c1)
#
#         x = self.block15_a(c4, c5)  # (1,512,32,32) #c5上采样，通道减半；c4不变；cat
#         x = self.block16_b(x)  # (1,256,32,32)#通道减半
#
#         x = self.block17_a(c3, x)  # (1,256,64,64)
#         x = self.block18_b(x)  # (1,128,64,64)
#
#         x = self.block19_a(c2, x)  # (1,128,128,128)
#         fea = self.block20_b(x)  # (1,64,128,128)
#
#         x = self.block21_a(c1, fea)  # (1,64,256,256)
#         x1 = self.block22_b(x)  # (1,32,256,256)
#
#         x = self.block23_c(x1)  # (1,16,256,256)
#         # x = x[:, 7:, :, :]
#
#         seg_x = self.block24_c(x1)
#         seg_x = self.block25_c(seg_x)
#
#         return x, torch.sigmoid(seg_x)
#
#
#
#
# if __name__ == '__main__':
#     model = MobileV2_MLSD_Large()
#     input = torch.randn(1, 3, 512, 512)
#     out = model(input)
#     print(out[0].shape)
#
#
