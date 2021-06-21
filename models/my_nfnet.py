from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import nfnet
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model

from models.cbam import CBAM


def _dcfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 
            'pool_size': (7, 7), 'crop_pct': 0.9, 'interpolation': 'bicubic',
            'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
            'first_conv': 'stem.conv1', 'classifier': 'head.fc', **kwargs}

default_cfgs = dict(
    nfnet_l0=_dcfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/nfnet_l0_ra2-45c6688d.pth',
        pool_size=(7, 7), input_size=(3, 224, 224), test_input_size=(3, 288, 288), crop_pct=1.0),
    my_nfnet_l0=_dcfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/nfnet_l0_ra2-45c6688d.pth',
        pool_size=(7, 7), input_size=(3, 224, 224), test_input_size=(3, 288, 288), crop_pct=1.0)
)


@dataclass
class NfCfg:
    depths: Tuple[int, int, int, int]
    channels: Tuple[int, int, int, int]
    alpha: float = 0.2
    stem_type: str = '3x3'
    stem_chs: Optional[int] = None
    group_size: Optional[int] = None
    attn_layer: Optional[str] = None
    attn_kwargs: dict = None
    attn_gain: float = 2.0  # NF correction gain to apply if attn layer is used
    width_factor: float = 1.0
    bottle_ratio: float = 0.5
    num_features: int = 0  # num out_channels for final conv, no final_conv if 0
    ch_div: int = 8  # round channels % 8 == 0 to keep tensor-core use optimal
    reg: bool = False  # enables EfficientNet-like options used in RegNet variants, expand from in_chs, se in middle
    extra_conv: bool = False  # extra 3x3 bottleneck convolution for NFNet models
    gamma_in_act: bool = False
    same_padding: bool = False
    skipinit: bool = False  # disabled by default, non-trivial performance impact
    zero_init_fc: bool = False
    act_layer: str = 'silu'


def _nfnet_cfg(
        depths, channels=(256, 512, 1536, 1536), group_size=128, bottle_ratio=0.5, feat_mult=2.,
        act_layer='gelu', attn_layer='se', attn_kwargs=None):
    num_features = int(channels[-1] * feat_mult)
    attn_kwargs = attn_kwargs if attn_kwargs is not None else dict(reduction_ratio=0.5, divisor=8)
    cfg = NfCfg(
        depths=depths, channels=channels, stem_type='deep_quad', stem_chs=128, group_size=group_size,
        bottle_ratio=bottle_ratio, extra_conv=True, num_features=num_features, act_layer=act_layer,
        attn_layer=attn_layer, attn_kwargs=attn_kwargs)
    return cfg

model_cfgs = dict(
    nfnet_l0=_nfnet_cfg(
        depths=(1, 2, 6, 3), feat_mult=1.5, group_size=64, bottle_ratio=0.25,
        attn_kwargs=dict(reduction_ratio=0.25, divisor=8), act_layer='silu'),
    my_nfnet_l0=_nfnet_cfg(
        depths=(1, 2, 6, 3), feat_mult=1.5, group_size=64, bottle_ratio=0.25,
        attn_kwargs=dict(reduction_ratio=0.25, divisor=8), act_layer='silu'),
)

def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0., .01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class My_NFNet_L0(nfnet.NormFreeNet):
    def __init__(self, cfg: NfCfg, num_classes=8, in_chans=3, drop_rate=.5, drop_path_rate=.25, use_cbam=False):
        super(My_NFNet_L0, self).__init__(cfg, num_classes, in_chans=in_chans,
                                          drop_rate=drop_rate, drop_path_rate=drop_path_rate)

        # Additional Modules
        self.num_features = 2304
        self.cbam = CBAM(self.num_features, reduction_ratio=16, bn=False) if use_cbam else None
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.alpha = nn.Sequential(nn.Linear(self.num_features, 1, bias=False), nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(self.num_features * 2, 1, bias=False), nn.Sigmoid())
        self.head = None
        self.classifier = nn.Linear(self.num_features * 2, num_classes)
        
        # Weight Initialization
        for m in self.modules():
            _weight_init(m)
        
    def forward_features(self, x):
        features = []
        alphas = []
        for i in range(6):
            f = x[:, :, :, :, i]

            f = super().forward_features(f)

            if self.cbam is not None:
                f = self.cbam(f)

            f = self.global_pool(f)
            f = f.view(f.size(0), -1)

            features.append(f)
            alphas.append(self.alpha(f))

        # Self-attention Module
        feature_stack = torch.stack(features, dim=2)
        alpha_stack = torch.stack(alphas, dim=2)
        alpha_stack = F.softmax(alpha_stack, dim=2)
        alpha_orig = alpha_stack[:, :, 0]
        alpha_part_max = alpha_stack[:, :, 1:].max(dim=2)[0]
        Fm = feature_stack.mul(alpha_stack).sum(2).div(alpha_stack.sum(2))
        
        # Relation-attention Module
        betas = []
        for i in range(len(features)):
            features[i] = torch.cat([features[i], Fm], dim=1)
            betas.append(self.beta(features[i]))
        
        feature_stack = torch.stack(features, dim=2)
        beta_stack = torch.stack(betas, dim=2)
        beta_stack = F.softmax(beta_stack, dim=2)
        output = (feature_stack.mul(beta_stack * alpha_stack)
                  .sum(2)
                  .div((beta_stack * alpha_stack).sum(2)))
        output = output.view(output.size(0), -1)
        
        return output, alpha_part_max, alpha_orig

    def forward(self, x):
        x, alpha_part_max, alpha_orig = self.forward_features(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        pred_score = self.classifier(x)

        return pred_score, alpha_part_max, alpha_orig


def _create_normfreenet(variant, pretrained=False, **kwargs):
    model_cfg = model_cfgs[variant]
    feature_cfg = dict(flatten_sequential=True)
    return build_model_with_cfg(
        My_NFNet_L0, variant, pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=model_cfg,
        feature_cfg=feature_cfg,
        **kwargs)
