"""
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/64645
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x

class DecoderHeng(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(DecoderHeng, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)  # False
        if e is not None:
            x = torch.cat([x, e], 1)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)  # False
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x


class UNetResNet34Heng(nn.Module):
    def __init__(self, dropout2d=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.dropout2d = dropout2d

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.decoder5 = DecoderHeng(256 + 512, 512, 64)
        self.decoder4 = DecoderHeng(64 + 256, 256, 64)
        self.decoder3 = DecoderHeng(64 + 128, 128, 64)
        self.decoder2 = DecoderHeng(64 + 64, 64, 64)
        self.decoder1 = DecoderHeng(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.conv1(x)

        e2 = self.encoder2(x)  # ; print('e2',e2.size())
        e3 = self.encoder3(e2)  # ; print('e3',e3.size())
        e4 = self.encoder4(e3)  # ; print('e4',e4.size())
        e5 = self.encoder5(e4)  # ; print('e5',e5.size())

        f = self.center(e5)
        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)

        f = torch.cat((
            d1,
            F.upsample(d2,scale_factor=2, mode='bilinear',align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False)
        ), 1)
        f = F.dropout2d(f, p=self.dropout2d)
        logit = self.logit(f).view(-1,1,128,128)
        # logit = torch.sigmoid(logit)
        return logit

        # f = self.decoder5(torch.cat([f, e5], 1))  # ; print('d5',f.size())
        # f = self.decoder4(torch.cat([f, e4], 1))  # ; print('d4',f.size())
        # f = self.decoder3(torch.cat([f, e3], 1))  # ; print('d3',f.size())
        # f = self.decoder2(torch.cat([f, e2], 1))  # ; print('d2',f.size())

        # logit = self.logit(f).view(-1, 128, 128)
        # logit = torch.sigmoid(logit)
        # return logit


class UNetResNet34(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(512 + 256, 512, 256)
        self.decoder4 = Decoder(256 + 256, 512, 256)
        self.decoder3 = Decoder(128 + 256, 256, 128)
        self.decoder2 = Decoder(64 + 128, 128, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)

        e2 = self.encoder2(x)  # ; print('e2',e2.size())
        e3 = self.encoder3(e2)  # ; print('e3',e3.size())
        e4 = self.encoder4(e3)  # ; print('e4',e4.size())
        e5 = self.encoder5(e4)  # ; print('e5',e5.size())

        # f = F.max_pool2d(e5, kernel_size=2, stride=2 )  #; print(f.size())
        # f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)#False
        # f = self.center(f)                       #; print('center',f.size())
        f = self.center(e5)

        f = self.decoder5(torch.cat([f, e5], 1))  # ; print('d5',f.size())
        f = self.decoder4(torch.cat([f, e4], 1))  # ; print('d4',f.size())
        f = self.decoder3(torch.cat([f, e3], 1))  # ; print('d3',f.size())
        f = self.decoder2(torch.cat([f, e2], 1))  # ; print('d2',f.size())

        logit = self.logit(f).view(-1,128,128)
        # logit = torch.sigmoid(logit)
        return logit