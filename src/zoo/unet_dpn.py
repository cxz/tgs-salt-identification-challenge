import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


from .dpn import *

class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)

class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_size=3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, conv_size, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class UnetDPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dpn = dpn131(pretrained=True)
        self.num_classes = 1

        #self.filters = [10, 144, 320, 704, 832]  # dpn68
        self.filters = [128, 352, 832, 1984, 2688] # dpn131
        self.encoder_stages = nn.ModuleList([self.get_encoder(idx) for idx in range(5)])

        self.bottleneck_type = ConvBottleneck
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(f * 2, f) for f in reversed(self.filters[:-1])])

        self.decoder_block = UnetDecoderBlock
        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(1, len(self.filters))])

        self.last_upsample = self.decoder_block(3296, 3296)

        #self.final = nn.Sequential(
        #    nn.Conv2d(self.filters[0], self.num_classes, 3, padding=1)
        #)

        self.final = nn.Sequential(
            nn.Conv2d(3296, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def get_decoder(self, layer):
        return self.decoder_block(self.filters[layer], self.filters[max(layer - 1, 0)])

    def get_encoder(self, layer):
        blocks = self.dpn.blocks

        if layer == 0:
            return nn.Sequential(
                blocks['conv1_1'].conv, #conv
                blocks['conv1_1'].bn, #bn
                blocks['conv1_1'].act, #relu
            )
        elif layer == 1:
            return nn.Sequential(
                blocks['conv1_1'].pool, #maxpool
                *[b for k, b in blocks.items() if k.startswith('conv2_')]
            )
        elif layer == 2:
            return nn.Sequential(*[b for k, b in blocks.items() if k.startswith('conv3_')])
        elif layer == 3:
            return nn.Sequential(*[b for k, b in blocks.items() if k.startswith('conv4_')])
        elif layer == 4:
            return nn.Sequential(*[b for k, b in blocks.items() if k.startswith('conv5_')])

    def forward(self, x):
        # print('input: ', x.size())
        # encoder
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        #for idx, r in enumerate(enc_results):
        #    print(f"encoder {idx} ", r.size())

        dec_results = []
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
            dec_results.append(x)

        # for r in dec_results:
        #     print(r.size())

        f = torch.cat((
            F.upsample(dec_results[3], scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(dec_results[2], scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(dec_results[1], scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(dec_results[0], scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)

        #f = self.last_upsample(f)
        f = self.final(f)
        return f


if __name__ == '__main__':
    print('.')