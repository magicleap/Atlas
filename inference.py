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

import argparse
import os

import numpy as np
import torch
import pytorch_lightning as pl

from atlas.data import SceneDataset, parse_splits_list
from atlas.model import VoxelNet
import atlas.transforms as transforms


def process(info_file, model, num_frames, save_path, total_scenes_index, total_scenes_count):
    """ Run the netork on a scene and save output

    Args:
        info_file: path to info_json file for the scene
        model: pytorch model that implemets Atlas
        frames: number of frames to use in reconstruction (-1 for all)
        save_path: where to save outputs
        total_scenes_index: used to print which scene we are on
        total_scenes_count: used to print the total number of scenes to process
    """

    voxel_scale = model.voxel_sizes[0]
    dataset = SceneDataset(info_file, voxel_sizes=[voxel_scale],
                           voxel_types=model.voxel_types, num_frames=num_frames)

    # compute voxel origin
    if 'file_name_vol_%02d'%voxel_scale in dataset.info:
        # compute voxel origin from ground truth
        tsdf_trgt = dataset.get_tsdf()['vol_%02d'%voxel_scale]
        voxel_size = float(voxel_scale)/100
        # shift by integer number of voxels for padding
        shift = torch.tensor([.5, .5, .5])//voxel_size
        offset = tsdf_trgt.origin - shift*voxel_size

    else:
        # use default origin
        # assume floor is a z=0 so pad bottom a bit
        offset = torch.tensor([0,0,-.5])
    T = torch.eye(4)
    T[:3,3] = offset

    transform = transforms.Compose([
        transforms.ResizeImage((640,480)),
        transforms.ToTensor(),
        transforms.TransformSpace(T, model.voxel_dim_val, [0,0,0]),
        transforms.IntrinsicsPoseToProjection(),
    ])
    dataset.transform = transform
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                             batch_sampler=None, num_workers=32)

    scene = dataset.info['scene']

    model.initialize_volume()
    torch.cuda.empty_cache()

    # trainer = pl.Trainer(
    #     distributed_backend='dp',
    #     benchmark=False,
    #     gpus=[5],
    #     precision=32)
    #     #num_sanitey_val_steps=0)

    # print(total_scenes_index,
    #       total_scenes_count,
    #       dataset.info['dataset'],
    #       scene,
    #       len(dataloader)
    # )

    # model.test_offset = offset.cuda()
    # model.save_path = save_path
    # model.scene = scene
    # trainer.test(model, test_dataloaders=dataloader)

    for j, d in enumerate(dataloader):

        # logging progress
        if j%25==0:
            print(total_scenes_index,
                  total_scenes_count,
                  dataset.info['dataset'],
                  scene,
                  j,
                  len(dataloader)
            )

        #print(d['projection'].unsqueeze(0).shape, d['image'].unsqueeze(0).shape)
        model.inference1(d['projection'].unsqueeze(0).cuda(),
                         image=d['image'].unsqueeze(0).cuda())

        outputs, losses = model.inference2()

        tsdf_pred = model.postprocess(outputs)[0]

        # TODO: set origin in model... make consistent with offset above?
        tsdf_pred.origin = offset.view(1,3).cuda()
    

        if 'semseg' in tsdf_pred.attribute_vols:
            mesh_pred = tsdf_pred.get_mesh('semseg')

            # save vertex attributes seperately since trimesh doesn't
            np.savez(os.path.join(save_path, '%s_attributes.npz'%scene), 
                    **mesh_pred.vertex_attributes)
        else:
            mesh_pred = tsdf_pred.get_mesh()

        tsdf_pred.save(os.path.join(save_path, '%s_%d.npz'%(scene,j)))
        mesh_pred.export(os.path.join(save_path, '%s_%d.ply'%(scene,j)))

    # outputs, losses = model.inference2()

    # tsdf_pred = model.postprocess(outputs)[0]

    # # TODO: set origin in model... make consistent with offset above?
    # tsdf_pred.origin = offset.view(1,3).cuda()
   

    # if 'semseg' in tsdf_pred.attribute_vols:
    #     mesh_pred = tsdf_pred.get_mesh('semseg')

    #     # save vertex attributes seperately since trimesh doesn't
    #     np.savez(os.path.join(save_path, '%s_attributes.npz'%scene), 
    #             **mesh_pred.vertex_attributes)
    # else:
    #     mesh_pred = tsdf_pred.get_mesh()

    # tsdf_pred.save(os.path.join(save_path, '%s.npz'%scene))
    # mesh_pred.export(os.path.join(save_path, '%s.ply'%scene))



def main():
    parser = argparse.ArgumentParser(description="Atlas Testing")
    parser.add_argument("--model", required=True, metavar="FILE",
                        help="path to checkpoint")
    parser.add_argument("--scenes", default="data/scannet_test.txt",
                        help="which scene(s) to run on")
    parser.add_argument("--num_frames", default=-1, type=int,
                        help="number of frames to use (-1 for all)")
    parser.add_argument("--voxel_dim", nargs=3, default=[-1,-1,-1], type=int,
                        help="override voxel dim")
    args = parser.parse_args()

    # get all the info_file.json's from the command line
    # .txt files contain a list of info_file.json's
    info_files = parse_splits_list(args.scenes)

    model = VoxelNet.load_from_checkpoint(args.model)
    model = model.cuda().eval()
    torch.set_grad_enabled(False)

    # overwrite default values of voxel_dim_test
    if args.voxel_dim[0] != -1:
        model.voxel_dim_test = args.voxel_dim
    # TODO: implement voxel_dim_test
    model.voxel_dim_val = model.voxel_dim_test

    model_name = os.path.splitext(os.path.split(args.model)[1])[0]
    save_path = os.path.join(model.cfg.LOG_DIR, model.cfg.TRAINER.NAME,
                             model.cfg.TRAINER.VERSION, 'test_'+model_name)
    if args.num_frames>-1:
        save_path = '%s_%d'%(save_path, args.num_frames)
    os.makedirs(save_path, exist_ok=True)

    for i, info_file in enumerate(info_files):
        # run model on each scene
        process(info_file, model, args.num_frames, save_path, i, len(info_files))

if __name__ == "__main__":
    main()