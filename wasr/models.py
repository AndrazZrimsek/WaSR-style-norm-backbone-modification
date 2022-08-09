from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import segmentation
from torchvision.models.resnet import resnet101, resnet50
from torch.hub import load_state_dict_from_url
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F

from .decoders import NoIMUDecoder, IMUDecoder
from .utils import IntermediateLayerGetter, replace_norm_layer

model_list = ['wasr_resnet101', 'wasr_resnet101_imu', 'wasr_resnet50', 'wasr_resnet50_imu', 'deeplab', 'wasr_resnet101_bin_imu']
model_urls = {
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
}

def get_model(model_name, num_classes=3, pretrained=True):
    imu = model_name.endswith('_imu')
    if model_name.startswith('wasr_resnet101'):
        if model_name.startswith('wasr_resnet101_bin'):
            model = wasr_deeplabv2_resnet101_bin(num_classes=num_classes, pretrained=pretrained, imu=imu)
        else:
            model = wasr_deeplabv2_resnet101(num_classes=num_classes, pretrained=pretrained, imu=imu)
    elif model_name.startswith('wasr_resnet50'):
        model = wasr_deeplabv2_resnet50(num_classes=num_classes, imu=imu)
    elif model_name == 'deeplab':
        model = deeplabv3_resnet101(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError('Unknown model: %s' % model_name)

    return model

class WaSR(nn.Module):
    """
    Implements WaSR model from
    `"A water-obstacle separation and refinement network for unmanned surface vehicles"
    <https://arxiv.org/abs/2001.01921>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the following keys:
            - "out": last feature map of the backbone (2048 features)
            - "aux": feature map used for the auxiliary separation loss (1024 features)
            - "skip1": high-resolution feature map (skip connection) used in FFM (256 features)
            - "skip2": low-resolution feature map (skip connection) used in ARM2 (512 features)
        decoder (nn.Module): a WaSR decoder module. Takes the backbone outputs (with skip connections)
            and returns a dense segmentation prediction for the classes
        classifier_input_features (int, optional): number of input features required by classifier
    """
    def __init__(self, backbone, decoder, imu=False):
        super(WaSR, self).__init__()

        self.imu = imu

        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        features = self.backbone(x['image'])

        if self.imu:
            features['imu_mask'] = x['imu_mask']

        x = self.decoder(features)

        # Return segmentation map and aux feature map
        output = OrderedDict([
            ('out', x),
            ('aux', features['aux'])
        ])

        return output


def wasr_deeplabv2_resnet101_bin(num_classes=3, pretrained=False, imu=True):
    # Pretrained ResNet101 backbone
    backbone = resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True])

    print("before replace------------------------")
    for name, param in model.layer1.named_parameters(): 
        if param.requires_grad: 
            print(name)

    norm_layer = BatchInstanceNorm2d
    replace_norm_layer(backbone.layer1, 'resnet101_layer1', 64, norm_layer)
    replace_norm_layer(backbone.layer2, 'resnet101_layer2', 128, norm_layer)
    replace_norm_layer(backbone.layer3, 'resnet101_layer3', 256, norm_layer)
    replace_norm_layer(backbone.layer4, 'resnet101_layer4', 512, norm_layer)

    print("after replace-------------------------")
    for name, param in model.layer1.named_parameters(): 
        if param.requires_grad: 
            print(name)

    return_layers = {
        'layer4': 'out',
        'layer1': 'skip1',
        'layer2': 'skip2',
        'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if imu:
        decoder = IMUDecoder(num_classes)
    else:
        decoder = NoIMUDecoder(num_classes)

    model = WaSR(backbone, decoder, imu=imu)

    # Load pretrained DeeplabV3 weights (COCO)
    if pretrained:
        model_url = model_urls['deeplabv3_resnet101_coco']
        state_dict = load_state_dict_from_url(model_url, progress=True)

        # Only load backbone weights, since decoder is entirely different, skip normalization layer because it has been changed
        keys_to_remove = [key for key in state_dict.keys() if not key.startswith('backbone.') or ('.bn1.' in key or '.bn2.' in key or '.bn3.' in key)]
        for key in keys_to_remove: del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

    print("after load--------------------------------")
    for name, param in model.layer1.named_parameters(): 
        if param.requires_grad: 
            print(name)

    return model


def wasr_deeplabv2_resnet101(num_classes=3, pretrained=False, imu=False):
    # Pretrained ResNet101 backbone
    backbone = resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True])
    return_layers = {
        'layer4': 'out',
        'layer1': 'skip1',
        'layer2': 'skip2',
        'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if imu:
        decoder = IMUDecoder(num_classes)
    else:
        decoder = NoIMUDecoder(num_classes)

    model = WaSR(backbone, decoder, imu=imu)

    # Load pretrained DeeplabV3 weights (COCO)
    if pretrained:
        model_url = model_urls['deeplabv3_resnet101_coco']
        state_dict = load_state_dict_from_url(model_url, progress=True)

        # Only load backbone weights, since decoder is entirely different
        keys_to_remove = [key for key in state_dict.keys() if not key.startswith('backbone.')]
        for key in keys_to_remove: del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

    return model


def wasr_deeplabv2_resnet50(num_classes=3, imu=False):
    # Pretrained ResNet101 backbone
    backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
    return_layers = {
        'layer4': 'out',
        'layer1': 'skip1',
        'layer2': 'skip2',
        'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if imu:
        decoder = IMUDecoder(num_classes)
    else:
        decoder = NoIMUDecoder(num_classes)

    model = WaSR(backbone, decoder, imu=imu)

    return model

class SegmentationNet(nn.Module):
    """Segmentation net wrapper for SOTA models."""
    def __init__(self, backbone, decoder):
        super(SegmentationNet, self).__init__()

        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        features = self.backbone(x['image'])

        x = self.decoder(features['out'])

        # Return segmentation map and aux feature map
        output = OrderedDict([
            ('out', x),
            ('aux', features['aux'])
        ])

        return output

def deeplabv3_resnet101(num_classes=3, pretrained=True):
    model = segmentation.deeplabv3_resnet101(pretrained=pretrained, aux_loss=False)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    backbone = model.backbone
    decoder = model.classifier

    return_layers = {
            'layer4': 'out',
            'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = SegmentationNet(backbone, decoder)

    return model


class _BatchInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, bn_w, self.bias,
            self.training, self.momentum, self.eps)
        
        # Instance norm
        b, c  = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in


class BatchInstanceNorm1d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm2d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm3d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))