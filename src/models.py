import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import other_models
import unet_resnet50

archs = [
    'unet-resnet50', 
    'unet-resnet101', 
    'unet-resnet152', 
    'unet-vgg11', 
    'unet-vgg16',
    'unet-dpn92',
    #'unet-dpn131',
    'unet-incv3',
    'unet-serefinenet101',
    'unet-serefinenet152'
]

def get_model(model_path, model_type):
    
    if model_type == 'unet-resnet101':
        model = UNetResNet(encoder_depth=101, num_classes=1, num_filters=16, dropout_2d=0.1, pretrained=True)

    elif model_type == 'unet-resnet152':
        model = UNetResNet(encoder_depth=152, num_classes=1, num_filters=16, dropout_2d=0.1, pretrained=True)

    elif model_type == 'unet-vgg11':
        model = other_models.UNet11()
        
    elif model_type == 'unet-vgg16':
        model = other_models.UNet16()
        
    elif model_type == 'unet-resnet50':
        model = unet_resnet50.UNet(num_classes=1, num_filters=32, pretrained=True)

    elif model_type == 'unet-dpn92':
        from zoo.albu_zoo.unet import DPNUnet
        model = DPNUnet(1, 3)

    elif model_type == 'unet-serefinenet101':
        from zoo.creafz.make_model import make_model
        # model = make_model('se_refinenet_128')
        model = make_model('se_refinenet_128_all_layers_unfrozen')        
        
    elif model_type == 'unet-serefinenet152':
        from zoo.creafz.make_model import make_model
        # model = make_model('se_refinenet_128_all_layers_unfrozen')
        # model = make_model('se_refinenet_128')
        raise # set encoder to resnet152

    #elif model_type == 'unet-dpn131':
    #    from zoo.albu_zoo.unet import DPNUnet
    #    model = DPNUnet(1, encoder_name='dpn131')

    elif model_type == 'unet-incv3':
        from zoo.albu_zoo.unet import Incv3
        model = Incv3(1, 3)

    else:
        raise NotImplementedError

    if model_path is not None:
        state = torch.load(str(model_path))
        state = {key.replace('module.', ''): value for key, value in state['model'].items()}
        model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()
    return model


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

    
class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)



class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.

    """

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))


if __name__ == '__main__':
    import numpy as np
    # model = UNetResNet(encoder_depth=101, num_classes=1, num_filters=16, dropout_2d=0.1, pretrained=True)
    model = UNetResNet(encoder_depth=101, num_classes=1, num_filters=16, dropout_2d=0.1, pretrained=True)
    # x1 = torch.from_numpy(np.zeros((32, 3, 128, 128), dtype=np.float32))
    # x2 = torch.from_numpy(np.zeros((32, 2), dtype=np.float32))
    # model.forward(x1, x2)
    # print('.')