import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)

from .resnet import BasicBlock


class BottomUpHigherResolutionHead(nn.Module):
    """Bottom-up head for Higher Resolution.
    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints
        tag_per_joint (bool): If tag_per_joint is True,
            the dimension of tags equals to num_joints,
            else the dimension of tags is 1. Default: True
        extra:
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        cat_output (list[bool]): Option to concat outputs.
        with_ae_loss (list[bool]): Option to use ae loss.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 tag_per_joint=True,
                 extra=None,
                 num_deconv_layers=1,
                 num_deconv_filters=(32,),
                 num_deconv_kernels=(4,),
                 num_basic_blocks=4,
                 cat_output=None,
                 with_ae_loss=None,
                 feature_fusion=None):
        super().__init__()

        dim_tag = num_joints if tag_per_joint else 1

        self.num_deconvs = num_deconv_layers
        self.cat_output = cat_output
        self.feature_fusion = feature_fusion

        final_layer_output_channels = []

        if with_ae_loss[0]:
            out_channels = num_joints + dim_tag
        else:
            out_channels = num_joints

        final_layer_output_channels.append(out_channels)
        for i in range(num_deconv_layers):
            if with_ae_loss[i + 1]:
                out_channels = num_joints + dim_tag
            else:
                out_channels = num_joints
            final_layer_output_channels.append(out_channels)

        deconv_layer_output_channels = []
        for i in range(num_deconv_layers):
            if with_ae_loss[i]:
                out_channels = num_joints + dim_tag
            else:
                out_channels = num_joints
            deconv_layer_output_channels.append(out_channels)

        self.final_layers = self._make_final_layers(
            in_channels, final_layer_output_channels, extra, num_deconv_layers,
            num_deconv_filters)
        self.deconv_layers = self._make_deconv_layers(
            in_channels, deconv_layer_output_channels, num_deconv_layers,
            num_deconv_filters, num_deconv_kernels, num_basic_blocks,
            cat_output)

    def _make_final_layers(self, in_channels, final_layer_output_channels,
                           extra, num_deconv_layers, num_deconv_filters):
        """Make final layers."""
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            else:
                padding = 0
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        final_layers = []
        final_layers.append(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=final_layer_output_channels[0],
                kernel_size=kernel_size,
                stride=1,
                padding=padding))

        for i in range(num_deconv_layers):
            in_channels = num_deconv_filters[i]
            final_layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=final_layer_output_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, in_channels, deconv_layer_output_channels,
                            num_deconv_layers, num_deconv_filters,
                            num_deconv_kernels, num_basic_blocks, cat_output):
        """Make deconv layers."""
        deconv_layers = []
        for i in range(num_deconv_layers):
            if cat_output[i]:
                in_channels += deconv_layer_output_channels[i]

            planes = num_deconv_filters[i]
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(num_deconv_kernels[i])

            layers = []
            layers.append(
                nn.Sequential(
                    build_upsample_layer(
                        dict(type='deconv'),
                        in_channels=in_channels,
                        out_channels=planes,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False), nn.BatchNorm2d(planes, momentum=0.1),
                    nn.ReLU(inplace=True)))
            for _ in range(num_basic_blocks):
                layers.append(nn.Sequential(BasicBlock(planes, planes), ))
            deconv_layers.append(nn.Sequential(*layers))
            in_channels = planes

        return nn.ModuleList(deconv_layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]

        final_outputs = []
        features_small = x
        y = self.final_layers[0](x)
        final_outputs.append(y)

        for i in range(self.num_deconvs):
            if self.cat_output[i]:
                x = torch.cat((x, y), 1)

            x = self.deconv_layers[i](x)
            y = self.final_layers[i + 1](x)
            final_outputs.append(y)

        features_big = x
        features_small = torch.nn.functional.interpolate(
            features_small,
            size=(features_big.shape[2], features_big.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        if self.feature_fusion == "pool":
            features = torch.max(features_small, features_big)
        elif self.feature_fusion =="avg":
            features = (features_big + features_small) / 2
        elif self.feature_fusion == "small":
            features = features_small
        elif self.feature_fusion == "large":
            features = features_big
        else:
            raise NotImplementedError

        return final_outputs, features

    def init_weights(self):
        """Initialize model weights."""
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for name, m in self.final_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)