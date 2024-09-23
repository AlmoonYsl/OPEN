import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.utils import Registry, build_from_cfg
from mmdet.models.backbones.resnet import BasicBlock

DepthNet = Registry('depthnet')


def build_depthnet(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, DepthNet, default_args)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            x: (B, C, H, W)
        """
        x1 = self.aspp1(x)      # (B, C, H, W)
        x2 = self.aspp2(x)      # (B, C, H, W)
        x3 = self.aspp3(x)      # (B, C, H, W)
        x4 = self.aspp4(x)      # (B, C, H, W)
        x5 = self.global_avg_pool(x)    # (B, C, 1, 1)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)      # (B, C, H, W)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)      # (B, 5*C, H, W)

        x = self.conv1(x)   # (B, C, H, W)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        """
        Args:
            x: (B, C, H, W)
            x_se: (B, C, 1, 1)
        Returns:

        """
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


@DepthNet.register_module()
class CameraAwareDepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels, num_params1=18,
                 num_params2=6, with_depth_correction=False, with_context_encoder=False, fix_alpha=None,
                 with_pgd=False):
        super(CameraAwareDepthNet, self).__init__()
        self.in_channels = in_channels
        self.depth_channels = depth_channels
        self.mid_channels = mid_channels
        self.fix_alpha = fix_alpha
        self.reduce_conv = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        )
        self.depth_mlp = Mlp(num_params2, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        if with_depth_correction:
            self.depth_stem = nn.Sequential(
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                ASPP(mid_channels, mid_channels),
            )
            self.depth_prob_conv = nn.Sequential(
                build_conv_layer(cfg=dict(
                    type='DCN',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    im2col_step=128,
                )),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, depth_channels,
                          kernel_size=1, stride=1, padding=0)
            )
        else:
            self.depth_stem = torch.nn.Identity()
            self.depth_prob_conv = nn.Conv2d(
                mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)

        self.with_pgd = with_pgd
        if self.with_pgd:
            if self.fix_alpha is not None:
                self.fuse_lambda = torch.tensor(self.fix_alpha).cuda()
            else:
                self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))

            self.depth_direct_conv = nn.Sequential(
                build_conv_layer(cfg=dict(
                    type='DCN',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    im2col_step=128,
                )),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x, intrinsics, extrinsics):
        """
        Args:
            x: img feature map  (B*N_view, C, H, W)
            intrinsics: (B, N_view, 3, 3)
            extrinsics: (B, N_view, 4, 4)
        Returns:
            depth:  (B*N_view, D, H, W)
            context: (B*N_view, C_context, H, W)
        """
        B, N_view = intrinsics.shape[:2]
        intrinsics = intrinsics[..., :2, :]  # 6
        mlp_input = intrinsics / 1e3
        x = self.reduce_conv(x)
        mlp_input = mlp_input.view(B * N_view, -1)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)  # (B*N_view, C_mid, H, W)
        if not self.with_pgd:
            depth_stem = self.depth_stem(depth)
            depth_prob = self.depth_prob_conv(
                depth_stem)  # (B*N_view, D, H, W)
            return depth_prob, x
        else:
            depth_stem = self.depth_stem(depth)
            depth_prob = self.depth_prob_conv(depth_stem)
            depth_direct = self.depth_direct_conv(depth_stem)

            return depth_prob, x, depth_direct

