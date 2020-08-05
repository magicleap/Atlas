# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

import torch
from torch import nn
from torch.nn import functional as F


class PixelHeads(nn.Module):
    """ Module that contains all the 2D output heads
    
    Features extracted by the 2D network are passed to this to produce 
    intermeditate per-frame outputs. Each type of output is added as a head 
    and is responsible for returning a dict of outputs and a dict of losses.
    """

    def __init__(self, cfg, stride):
        super().__init__()
        self.heads = nn.ModuleList()

        if "semseg" in cfg.MODEL.HEADS2D.HEADS:
            self.heads.append(SemSegHead(cfg, stride))

    def forward(self, x, targets=None):
        outputs = {}
        losses = {}

        for head in self.heads:
            out, loss = head(x, targets)
            outputs = { **outputs, **out }
            losses = { **losses, **loss }

        return outputs, losses



class SemSegHead(nn.Module):
    """ 2D image semantic segmentation"""

    def __init__(self, cfg, stride):
        super().__init__()

        self.loss_weight = cfg.MODEL.HEADS2D.SEMSEG.LOSS_WEIGHT
        channels_in = cfg.MODEL.BACKBONE3D.CHANNELS[0]
        self.stride = stride
        self.decoder = nn.Conv2d(channels_in,
                                 cfg.MODEL.HEADS2D.SEMSEG.NUM_CLASSES,
                                 1, bias=False)

    def forward(self, x, targets=None):
        output = {}
        losses = {}

        output['semseg'] = F.interpolate(self.decoder(x), 
                                         scale_factor=self.stride)

        if targets is not None and 'semseg' in targets:
            losses['semseg'] = F.cross_entropy(
                output['semseg'], targets['semseg'], ignore_index=-1
            ) * self.loss_weight

        return output, losses





