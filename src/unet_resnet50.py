'''U-Net inspired encoder-decoder architecture with a ResNet encoder as proposed by Alexander Buslaev.

See:
- https://arxiv.org/abs/1505.04597 - U-Net: Convolutional Networks for Biomedical Image Segmentation
- https://arxiv.org/abs/1411.4038  - Fully Convolutional Networks for Semantic Segmentation
- https://arxiv.org/abs/1512.03385 - Deep Residual Learning for Image Recognition
- https://arxiv.org/abs/1801.05746 - TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
- https://arxiv.org/abs/1806.00844 - TernausNetV2: Fully Convolutional Network for Instance Segmentation

copied from https://raw.githubusercontent.com/mapbox/robosat/6f9957dc05420573f42948bae8956fdbc5f82bd5/robosat/unet.py
'''

import torch
import torch.nn as nn

from torchvision.models import resnet50


def conv3x3(num_in, num_out):
    '''Creates a 3x3 convolution building block module.

    Args:
      num_in: number of input feature maps
      num_out: number of output feature maps

    Returns:
      The 3x3 convolution module.
    '''

    return nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)


class ConvRelu(nn.Module):
    '''Convolution followed by ReLU activation building block.
    '''

    def __init__(self, num_in, num_out):
        '''Creates a `ConvReLU` building block.

        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        '''

        super().__init__()

        self.block = nn.Sequential(
            conv3x3(num_in, num_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        '''The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        '''

        return self.block(x)


class DecoderBlock(nn.Module):
    '''Decoder building block upsampling resolution by a factor of two.
    '''

    def __init__(self, num_in, num_out):
        '''Creates a `DecoderBlock` building block.

        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        '''

        super().__init__()

        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        '''The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        '''

        return self.block(nn.functional.upsample(x, scale_factor=2, mode='nearest'))


class UNet(nn.Module):
    '''The "U-Net" architecture for semantic segmentation, adapted by changing the encoder to a ResNet feature extractor.

       Also known as AlbuNet due to its inventor Alexander Buslaev.
    '''

    def __init__(self, num_classes, num_filters=32, pretrained=True):
        '''Creates an `UNet` instance for semantic segmentation.

        Args:
          num_classes: number of classes to predict.
          pretrained: use ImageNet pre-trained backbone feature extractor
        '''

        super().__init__()

        self.resnet = resnet50(pretrained=pretrained)

        self.enc0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.enc1 = self.resnet.layer1 #  256
        self.enc2 = self.resnet.layer2 #  512
        self.enc3 = self.resnet.layer3 # 1024
        self.enc4 = self.resnet.layer4 # 2048

        self.center = DecoderBlock(2048, num_filters * 8)

        self.dec0 = DecoderBlock(2048 + num_filters * 8, num_filters * 8)
        self.dec1 = DecoderBlock(1024 + num_filters * 8, num_filters * 8)
        self.dec2 = DecoderBlock(512 + num_filters * 8, num_filters * 2)
        self.dec3 = DecoderBlock(256 + num_filters * 2, num_filters * 2 * 2)
        self.dec4 = DecoderBlock(num_filters * 2 * 2, num_filters)
        self.dec5 = ConvRelu(num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)        

    def forward(self, x):
        '''The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        '''

        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        center = self.center(nn.functional.max_pool2d(enc4, kernel_size=2, stride=2))

        dec0 = self.dec0(torch.cat([enc4, center], dim=1))
        dec1 = self.dec1(torch.cat([enc3, dec0], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc1, dec2], dim=1))
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)

        return self.final(dec5)
    
