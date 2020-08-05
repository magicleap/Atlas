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

import os

import pytorch_lightning as pl
import torch

from atlas.config import get_parser, get_cfg
from atlas.logger import AtlasLogger
from atlas.model import VoxelNet


# FIXME: should not be necessary, but something is remaining
# in memory between train and val
class CudaClearCacheCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_end(self, trainer, pl_module):
        torch.cuda.empty_cache()


if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = get_cfg(args)
    model = VoxelNet(cfg.convert_to_dict())

    save_path = os.path.join(cfg.LOG_DIR, cfg.TRAINER.NAME, cfg.TRAINER.VERSION)
    logger = AtlasLogger(cfg.LOG_DIR, cfg.TRAINER.NAME, cfg.TRAINER.VERSION)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path, '{epoch:03d}'),
        save_top_k=-1,
        period=cfg.TRAINER.CHECKPOINT_PERIOD)

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=cfg.TRAINER.CHECKPOINT_PERIOD,
        callbacks=[CudaClearCacheCallback()],
        distributed_backend='ddp',
        benchmark=True,
        gpus=cfg.TRAINER.NUM_GPUS,
        precision=cfg.TRAINER.PRECISION,
        amp_level='O1')
    
    trainer.fit(model)

