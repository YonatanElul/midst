# The implementation for the 1D-ResNet model was heavily based on https://github.com/hsd1503/resnet1d
# The implementation for the 2D-ResNet model was heavily based on PyTorch's official implementation
# at https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
from midst.models.fc_models import MCDropout, get_activation_layer
from midst.utils.defaults import MODELS_TENSOR_PREDICITONS_KEY, OTHER_KEY
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        dim: int = 2,
) -> Union[nn.Conv1d, nn.Conv2d]:
    """1x1 convolution"""
    if dim == 1:
        return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    elif dim == 2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        dim: int = 2,
) -> Union[nn.Conv1d, nn.Conv2d]:
    """3x3 convolution with padding"""
    if dim == 1:
        return nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )

    elif dim == 2:
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dim: int = 2,
            activation: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            if dim == 1:
                norm_layer = nn.BatchNorm1d

            elif dim == 2:
                norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dim=dim)
        self.bn1 = norm_layer(planes)

        if activation is None:
            self.activation = nn.ReLU(inplace=True)

        else:
            self.activation = activation

        self.conv2 = conv3x3(planes, planes, dim=dim)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation: nn.Module = None,
            dropout: float = 0,
            mc_dropout: bool = False,
            dim: int = 2,
    ) -> None:
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, dim=dim)
        self._norm_layer = norm_layer
        if norm_layer is not None:
            self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation, dim=dim)
        if norm_layer is not None:
            self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion, dim=dim)
        if norm_layer is not None:
            self.bn3 = norm_layer(planes * self.expansion)

        if dropout:
            if mc_dropout:
                self.dropout = MCDropout(dropout)

            else:
                self.dropout = nn.Dropout(dropout)

        else:
            self.dropout = None

        if activation is None:
            self.activation = nn.ReLU(inplace=True)

        else:
            self.activation = activation

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        if self._norm_layer is not None:
            out = self.bn1(out)

        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        if self._norm_layer is not None:
            out = self.bn2(out)

        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv3(out)
        if self._norm_layer is not None:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class DeconvBottleneck1D(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            upsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation: nn.Module = nn.ReLU(inplace=True),
            dropout: float = 0,
            mc_dropout: bool = False,
    ) -> None:
        super(DeconvBottleneck1D, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        self.dconv1 = nn.ConvTranspose1d(
            inplanes,
            width,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
            dilation=1,
        )
        self._norm_layer = norm_layer
        if norm_layer is not None:
            self.bn1 = norm_layer(width)

        self.dconv2 = nn.ConvTranspose1d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=0,
            groups=1,
            bias=True,
            dilation=dilation,
        )
        if norm_layer is not None:
            self.bn2 = norm_layer(width)

        self.dconv3 = nn.ConvTranspose1d(
            width,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
            dilation=1,
        )
        if norm_layer is not None:
            self.bn3 = norm_layer(planes * self.expansion)

        if dropout:
            if mc_dropout:
                self.dropout = MCDropout(dropout)

            else:
                self.dropout = nn.Dropout(dropout)

        else:
            self.dropout = None

        self.activation = activation
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.dconv1(x)
        if self._norm_layer is not None:
            out = self.bn1(out)

        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.dconv2(out)
        if self._norm_layer is not None:
            out = self.bn2(out)

        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.dconv3(out)
        if self._norm_layer is not None:
            out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out = out + identity[..., 0][..., None]
        out = self.activation(out)

        return out


class ResNet2D(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            prediction_horizon: int = 1,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation: Optional[Union[str, dict]] = None,
    ) -> None:
        super(ResNet2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self._prediction_horizon = prediction_horizon
        self._activation_type = activation
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            input_dim,
            self.inplanes,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)

        if activation is not None:
            self.activation = get_activation_layer(activation)

        else:
            self.activation = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            activation=(get_activation_layer(activation) if activation is not None else None)
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            activation=(get_activation_layer(activation) if activation is not None else None)
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            activation=(get_activation_layer(activation) if activation is not None else None)
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            activation=(get_activation_layer(activation) if activation is not None else None)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            activation: Optional[nn.Module] = None,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
                activation=activation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    activation=activation,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Dict:
        x_shape = x.shape

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        predictions = x.reshape(x_shape[0], x_shape[1], self._prediction_horizon, -1)

        output = {
            MODELS_TENSOR_PREDICITONS_KEY: predictions,
            OTHER_KEY: {
            },
        }

        return output

    def forward(self, x: Tensor) -> Dict:
        return self._forward_impl(x)


class MultiClassResNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            n_classes: int,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(MultiClassResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self._norm_layer = norm_layer
        self._n_classes = n_classes
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(
            input_dim,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1, ))
        self.fc = nn.ModuleList(
            [
                nn.Linear(512 * block.expansion, 2)
                for _ in range(n_classes)
            ]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, dim=1),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                dim=1,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    dim=1,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Dict:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        classifications = [
            self.fc[i](x[..., 0])[None, ]
            for i in range(len(self.fc))
        ]
        classifications = torch.cat(classifications, dim=0)

        output = {
            MODELS_TENSOR_PREDICITONS_KEY: classifications,
            OTHER_KEY: {
            },
        }

        return output

    def forward(self, x: Tensor) -> Dict:
        return self._forward_impl(x)


def _resnet(
        input_dim: int,
        output_dim: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        prediction_horizon: int = 1,
        **kwargs: Any
) -> ResNet2D:
    model = ResNet2D(
        input_dim=input_dim,
        output_dim=output_dim,
        block=block,
        layers=layers,
        prediction_horizon=prediction_horizon,
        **kwargs,
    )
    return model


def resnet18(
        input_dim: int,
        output_dim: int,
        prediction_horizon: int = 1,
        **kwargs: Any,
) -> ResNet2D:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(input_dim, output_dim, BasicBlock, [2, 2, 2, 2], prediction_horizon, **kwargs)


def resnet34(
        input_dim: int,
        output_dim: int,
        prediction_horizon: int = 1,
        **kwargs: Any,
) -> ResNet2D:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(input_dim, output_dim, BasicBlock, [3, 4, 6, 3], prediction_horizon, **kwargs)


def resnet50(input_dim: int,
             output_dim: int,
             prediction_horizon: int = 1,
             **kwargs: Any,
             ) -> ResNet2D:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(input_dim, output_dim, Bottleneck, [3, 4, 6, 3], prediction_horizon, **kwargs)


def resnet101(input_dim: int,
              output_dim: int,
              prediction_horizon: int = 1,
              **kwargs: Any,
              ) -> ResNet2D:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(input_dim, output_dim, Bottleneck, [3, 4, 23, 3], prediction_horizon, **kwargs)


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()

        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock1D(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            downsample,
            use_bn,
            use_do,
            is_first_block=False,
    ):
        super(BasicBlock1D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample

        if self.downsample:
            self.stride = stride

        else:
            self.stride = 1

        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)

            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)

        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)

        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)

        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class ResNet1D(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    """

    def __init__(
            self,
            in_channels: int,
            n_filters_first_layer: int,
            kernel_size: int,
            stride: int,
            groups: int,
            n_block: int,
            output_dim: int,
            prediction_steps: int,
            downsample_gap: int = 2,
            increasefilter_gap: int = 4,
            use_bn: bool = True,
            use_do: bool = True,
    ):
        """

        :param in_channels: Numer of input channels
        :param n_filters_first_layer: Number of units in the first layer
        :param kernel_size: kernel size across layers
        :param stride: stride across layers
        :param groups: Number of groups
        :param n_block: Number of blocks
        :param output_dim: output dimensionality
        :param prediction_steps: Prediction horizon
        :param downsample_gap: Downsampling rate
        :param increasefilter_gap: Downsampling rate when increasing kernel size
        :param use_bn: Whether to use BatchNorm or not
        :param use_do: Whether to use Dropout or not
        """

        super(ResNet1D, self).__init__()

        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.prediction_steps = prediction_steps

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=n_filters_first_layer,
                                                kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(n_filters_first_layer)
        self.first_block_relu = nn.ReLU()
        out_channels = n_filters_first_layer

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True

            else:
                is_first_block = False

            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True

            else:
                downsample = False

            # in_channels and out_channels
            if is_first_block:
                in_channels = n_filters_first_layer
                out_channels = in_channels

            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(n_filters_first_layer * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block,
            )
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, output_dim)

    def forward(self, x) -> Dict:
        # first conv
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)

        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            out = net(out)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)

        out = self.final_relu(out)
        out = out.mean(-1)
        out = self.dense(out)

        predictions = out.reshape(x.shape[0], self.prediction_steps, x.shape[3])

        output = {
            MODELS_TENSOR_PREDICITONS_KEY: predictions,
            OTHER_KEY: {
            },
        }

        return output
