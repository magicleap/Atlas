# from Detectron2: (https://github.com/facebookresearch/detectron2)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, get_norm
from detectron2.modeling.backbone import build_backbone as d2_build_backbone
import fvcore.nn.weight_init as weight_init



def build_backbone2d(cfg):
    """ Builds 2D feature extractor backbone network from Detectron2."""

    output_dim = cfg.MODEL.BACKBONE3D.CHANNELS[0]
    norm = cfg.MODEL.FPN.NORM
    output_stride = 4  # TODO: make configurable

    backbone = d2_build_backbone(cfg)
    feature_extractor = FPNFeature(
        backbone.output_shape(), output_dim, output_stride, norm)

    # load pretrained backbone
    if cfg.MODEL.BACKBONE.WEIGHTS:
        state_dict = torch.load(cfg.MODEL.BACKBONE.WEIGHTS)
        backbone.load_state_dict(state_dict)

    return nn.Sequential(backbone, feature_extractor), output_stride


class FPNFeature(nn.Module):
    """ Converts feature pyrimid to singe feature map (from Detectron2)"""
    
    def __init__(self, input_shape, output_dim=32, output_stride=4, norm='BN'):
        super().__init__()

        # fmt: off
        self.in_features      = ["p2", "p3", "p4", "p5"]
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(output_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else output_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, output_dim),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != output_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

    def forward(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        return x
