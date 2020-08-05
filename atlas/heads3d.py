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

from atlas.backbone3d import get_norm_3d


class VoxelHeads(nn.Module):
    """ Module that contains all the 3D output heads
    
    Features extracted by the 3D network are passed to this to produce the
    final outputs. Each type of output is added as a head and is responsible
    for returning a dict of outputs and a dict of losses
    """

    def __init__(self, cfg):
        super().__init__()
        self.heads = nn.ModuleList()

        if "tsdf" in cfg.MODEL.HEADS3D.HEADS:
            self.heads.append(TSDFHead(cfg))

        if "semseg" in cfg.MODEL.HEADS3D.HEADS:
            self.heads.append(SemSegHead(cfg))

        if "color" in cfg.MODEL.HEADS3D.HEADS:
            self.heads.append(ColorHead(cfg))


    def forward(self, x, targets=None):
        outputs = {}
        losses = {}

        for head in self.heads:
            out, loss = head(x, targets)
            outputs = { **outputs, **out }
            losses = { **losses, **loss }

        return outputs, losses


class TSDFHead(nn.Module):
    """ Main head that regresses the TSDF"""

    def __init__(self, cfg):
        super().__init__()

        self.loss_weight = cfg.MODEL.HEADS3D.TSDF.LOSS_WEIGHT

        self.label_smoothing = cfg.MODEL.HEADS3D.TSDF.LABEL_SMOOTHING

        self.multi_scale = cfg.MODEL.HEADS3D.MULTI_SCALE
        self.split_loss = cfg.MODEL.HEADS3D.TSDF.LOSS_SPLIT
        self.log_transform_loss = cfg.MODEL.HEADS3D.TSDF.LOSS_LOG_TRANSFORM
        self.log_transform_loss_shift = cfg.MODEL.HEADS3D.TSDF.LOSS_LOG_TRANSFORM_SHIFT
        self.sparse_threshold = cfg.MODEL.HEADS3D.TSDF.SPARSE_THRESHOLD

        scales = len(cfg.MODEL.BACKBONE3D.CHANNELS)-1
        final_size = int(cfg.VOXEL_SIZE*100)

        if self.multi_scale:
            self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
            decoders = [nn.Conv3d(c, 1, 1, bias=False) 
                        for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1]
        else:
            self.voxel_sizes = [final_size]
            decoders = [nn.Conv3d(cfg.MODEL.BACKBONE3D.CHANNELS[0], 1, 1, bias=False)]


        self.decoders = nn.ModuleList(decoders)


    def forward(self, xs, targets=None):
        output = {}
        losses = {}
        mask_surface_pred = []

        if not self.multi_scale:
            xs = xs[-1:]

        for i, (decoder, x) in enumerate(zip(self.decoders, xs)):
            # regress the TSDF
            tsdf = torch.tanh(decoder(x)) * self.label_smoothing

            # use previous scale to sparsify current scale
            if self.split_loss=='pred' and i>0:
                tsdf_prev = output['vol_%02d_tsdf'%self.voxel_sizes[i-1]]
                tsdf_prev = F.interpolate(tsdf_prev, scale_factor=2)
                # FIXME: when using float16, why is interpolate casting to float32?
                tsdf_prev = tsdf_prev.type_as(tsdf)
                mask_surface_pred_prev = tsdf_prev.abs()<self.sparse_threshold[i-1]
                # .999 so we don't close surfaces during mc
                tsdf[~mask_surface_pred_prev] = tsdf_prev[~mask_surface_pred_prev].sign()*.999
                mask_surface_pred.append( mask_surface_pred_prev )

            output['vol_%02d_tsdf'%self.voxel_sizes[i]] = tsdf

        # compute losses
        if targets is not None:
            for i, voxel_size in enumerate(self.voxel_sizes):
                key = 'vol_%02d_tsdf'%voxel_size
                pred = output[key]
                trgt = targets[key]

                mask_observed = trgt<1
                mask_outside  = (trgt==1).all(-1, keepdim=True)

                # TODO: extend mask_outside (in heads:loss) to also include 
                # below floor... maybe modify padding_mode in tsdf.transform... 
                # probably cleaner to look along slices similar to how we look
                # along columns for outside.

                if self.log_transform_loss:
                    pred = log_transform(pred, self.log_transform_loss_shift)
                    trgt = log_transform(trgt, self.log_transform_loss_shift)

                loss = F.l1_loss(pred, trgt, reduction='none') * self.loss_weight

                if self.split_loss=='none':
                    losses[key] = loss[mask_observed | mask_outside].mean()

                elif self.split_loss=='pred':
                    if i==0:
                        # no sparsifing mask for first resolution
                        losses[key] = loss[mask_observed | mask_outside].mean()
                    else:
                        mask = mask_surface_pred[i-1] & (mask_observed | mask_outside)
                        if mask.sum()>0:
                            losses[key] = loss[mask].mean()
                        else:
                            losses[key] = 0*loss.sum()

                else:
                    raise NotImplementedError("TSDF loss split [%s] not supported"%self.split_loss_empty)

        return output, losses

def log_transform(x, shift=1):
    """ rescales TSDF values to weight voxels near the surface more than close
    to the truncation distance"""
    return x.sign() * (1 + x.abs() / shift).log()


class SemSegHead(nn.Module):
    """ Predicts voxel semantic segmentation"""

    def __init__(self, cfg):
        super().__init__()

        self.multi_scale = cfg.MODEL.HEADS3D.MULTI_SCALE
        self.loss_weight = cfg.MODEL.HEADS3D.SEMSEG.LOSS_WEIGHT

        scales = len(cfg.MODEL.BACKBONE3D.CHANNELS)-1
        final_size = int(cfg.VOXEL_SIZE*100)

        classes = cfg.MODEL.HEADS3D.SEMSEG.NUM_CLASSES
        if self.multi_scale:
            self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
            decoders = [nn.Conv3d(c, classes, 1, bias=False) 
                        for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1]
        else:
            self.voxel_sizes = [final_size]
            decoders = [nn.Conv3d(cfg.MODEL.BACKBONE3D.CHANNELS[0], classes, 1, bias=False)]

        self.decoders = nn.ModuleList(decoders)

    def forward(self, xs, targets=None):
        output = {}
        losses = {}

        if not self.multi_scale:
            xs = xs[-1:] # just use final scale

        for voxel_size, decoder, x in zip(self.voxel_sizes, self.decoders, xs):
            # compute semantic labels
            key = 'vol_%02d_semseg'%voxel_size
            output[key] = decoder(x)

            # compute losses
            if targets is not None and key in targets:
                pred = output[key]
                trgt = targets[key]
                mask_surface = targets['vol_%02d_tsdf'%voxel_size].squeeze(1).abs() < 1

                loss = F.cross_entropy(pred, trgt, reduction='none', ignore_index=-1)
                if mask_surface.sum()>0:
                    loss = loss[mask_surface].mean()
                else:
                    loss = 0 * loss.mean()
                losses[key] = loss * self.loss_weight

        return output, losses


class ColorHead(nn.Module):
    """ Predicts voxel color"""

    def __init__(self, cfg):
        super().__init__()

        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1, 1)

        self.multi_scale = cfg.MODEL.HEADS3D.MULTI_SCALE
        self.loss_weight = cfg.MODEL.HEADS3D.COLOR.LOSS_WEIGHT

        scales = len(cfg.MODEL.BACKBONE3D.CHANNELS)-1
        final_size = int(cfg.VOXEL_SIZE*100)

        if self.multi_scale:
            self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
            decoders = [nn.Conv3d(c, 3, 1, bias=False) 
                        for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1]
        else:
            self.voxel_sizes = [final_size]
            decoders = [nn.Conv3d(cfg.MODEL.BACKBONE3D.CHANNELS[0], 3, 1, bias=False)]

        self.decoders = nn.ModuleList(decoders)

    def forward(self, xs, targets=None):
        output = {}
        losses = {}

        if not self.multi_scale:
            xs = xs[-1:] # just use final scale

        for voxel_size, decoder, x in zip(self.voxel_sizes, self.decoders, xs):
            key = 'vol_%02d_color'%voxel_size
            pred = torch.sigmoid(decoder(x)) * 255
            output[key] = pred

            # compute losses
            if targets is not None and key in targets:
                pred = output[key]
                trgt = targets[key]
                mask_surface = targets['vol_%02d_tsdf'%voxel_size].squeeze(1).abs() < 1

                loss = F.l1_loss(pred, trgt, reduction='none').mean(1)
                if mask_surface.sum()>0:
                    loss = loss[mask_surface].mean()
                else:
                    loss = 0 * loss.mean()
                losses[key] = loss * self.loss_weight / 255

        return output, losses




