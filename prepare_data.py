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
import json
import os

import open3d as o3d
import numpy as np
import torch
import trimesh

from atlas.data import SceneDataset, load_info_json
from atlas.datasets.scannet import prepare_scannet_scene, prepare_scannet_splits
from atlas.datasets.rio import prepare_rio_scene
from atlas.datasets.sample import prepare_sample_scene
import atlas.transforms as transforms
from atlas.tsdf import TSDFFusion, TSDF, coordinates, depth_to_world


def fuse_scene(path_meta, scene, voxel_size, trunc_ratio=3, max_depth=3,
               vol_prcnt=.995, vol_margin=1.5, fuse_semseg=False, device=0,
               verbose=2):
    """ Use TSDF fusion with GT depth maps to generate GT TSDFs

    Args:
        path_meta: path to save the TSDFs 
            (we recommend creating a parallel directory structure to save 
            derived data so that we don't modify the original dataset)
        scene: name of scene to process
        voxel_size: voxel size of TSDF
        trunc_ratio: truncation distance in voxel units
        max_depth: mask out large depth values since they are noisy
        vol_prcnt: for computing the bounding volume of the TSDF... ignore outliers
        vol_margin: padding for computing bounding volume of the TSDF
        fuse_semseg: whether to accumulate semseg images for GT semseg
            (prefered method is to not accumulate and insted transfer labels
            from ground truth labeled mesh)
        device: cpu/ which gpu
        verbose: how much logging to print

    Returns:
        writes a TSDF (.npz) file into path_meta/scene

    Notes: we use a conservative value of max_depth=3 to reduce noise in the 
    ground truth. However, this means some distant data is missing which can
    create artifacts. Nevertheless, we found we acheived the best 2d metrics 
    with the less noisy ground truth.
    """

    if verbose>0:
        print('fusing', scene, 'voxel size', voxel_size)

    info_file = os.path.join(path_meta, scene, 'info.json')

    # get gpu device for this worker
    device = torch.device('cuda', device) # gpu for this process

    # get the dataset
    transform = transforms.Compose([transforms.ResizeImage((640,480)),
                                    transforms.ToTensor(),
                                    transforms.InstanceToSemseg('nyu40'),
                                    transforms.IntrinsicsPoseToProjection(),
                                  ])
    frame_types=['depth', 'semseg'] if fuse_semseg else ['depth']
    dataset = SceneDataset(info_file, transform, frame_types)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                             batch_sampler=None, num_workers=4)

    # find volume bounds and origin by backprojecting depth maps to point clouds
    # use a subset of the frames to save time
    if len(dataset)<=200:
        dataset1 = dataset
    else:
        inds = np.linspace(0,len(dataset)-1,200).astype(np.int)
        dataset1 = torch.utils.data.Subset(dataset, inds)
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=None,
                                              batch_sampler=None, num_workers=4)

    pts = []
    for i, frame in enumerate(dataloader1):
        projection = frame['projection'].to(device)
        depth = frame['depth'].to(device)
        depth[depth>max_depth]=0
        pts.append( depth_to_world(projection, depth).view(3,-1).T )
    pts = torch.cat(pts)
    pts = pts[torch.isfinite(pts[:,0])].cpu().numpy()
    # use top and bottom vol_prcnt of points plus vol_margin
    origin = torch.as_tensor(np.quantile(pts, 1-vol_prcnt, axis=0)-vol_margin).float()
    vol_max = torch.as_tensor(np.quantile(pts, vol_prcnt, axis=0)+vol_margin).float()
    vol_dim = ((vol_max-origin)/(float(voxel_size)/100)).int().tolist()


    # initialize tsdf
    tsdf_fusion = TSDFFusion(vol_dim, float(voxel_size)/100, origin,
                             trunc_ratio, device, label=fuse_semseg)

    # integrate frames
    for i, frame in enumerate(dataloader):
        if verbose>1 and i%25==0:
            print(scene, 'integrating voxel size', voxel_size, i, len(dataset))

        projection = frame['projection'].to(device)
        image = frame['image'].to(device)
        depth = frame['depth'].to(device)
        semseg = frame['semseg'].to(device) if fuse_semseg else None

        # only use reliable depth
        depth[depth>max_depth]=0

        tsdf_fusion.integrate(projection, depth, image, semseg)

    # save mesh and tsdf
    file_name_vol = os.path.join(path_meta, scene, 'tsdf_%02d.npz'%voxel_size)
    file_name_mesh = os.path.join(path_meta, scene, 'mesh_%02d.ply'%voxel_size)
    tsdf = tsdf_fusion.get_tsdf()
    tsdf.save(file_name_vol)
    mesh = tsdf.get_mesh()
    mesh.export(file_name_mesh)
    if fuse_semseg:
        mesh = tsdf.get_mesh('instance')
        mesh.export(file_name_mesh.replace('.ply','_semseg.ply'))

    # update info json
    data = load_info_json(info_file)
    data['file_name_vol_%02d'%voxel_size] = file_name_vol
    json.dump(data, open(info_file, 'w'))


# use labeled mesh to label surface voxels in tsdf
def label_scene(path_meta, scene, voxel_size, dist_thresh=.05, verbose=2):
    """ Transfer instance labels from ground truth mesh to TSDF

    For each voxel find the nearest vertex and transfer the label if
    it is close enough to the voxel.

    Args:
        path_meta: path to save the TSDFs 
            (we recommend creating a parallel directory structure to save 
            derived data so that we don't modify the original dataset)
        scene: name of scene to process
        voxel_size: voxel size of TSDF to process
        dist_thresh: beyond this distance labels are not transferd
        verbose: how much logging to print

    Returns:
        Updates the TSDF (.npz) file with the instance volume
    """

    # dist_thresh: beyond this distance to nearest gt mesh vertex, 
    # voxels are not labeled
    if verbose>0:
        print('labeling', scene)

    info_file = os.path.join(path_meta, scene, 'info.json')
    data = load_info_json(info_file)

    # each vertex in gt mesh indexs a seg group
    segIndices = json.load(open(data['file_name_seg_indices'], 'r'))['segIndices']

    # maps seg groups to instances
    segGroups = json.load(open(data['file_name_seg_groups'], 'r'))['segGroups']
    mapping = {ind:group['id']+1 for group in segGroups for ind in group['segments']}

    # get per vertex instance ids (0 is unknown, [1,...] are objects)
    n = len(segIndices)
    instance_verts = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        if segIndices[i] in mapping:
            instance_verts[i] = mapping[segIndices[i]]

    # load vertex locations
    mesh = trimesh.load(data['file_name_mesh_gt'], process=False)
    verts = mesh.vertices

    # construct kdtree of vertices for fast nn lookup
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(verts)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # load tsdf volume
    tsdf = TSDF.load(data['file_name_vol_%02d'%voxel_size])
    coords = coordinates(tsdf.tsdf_vol.size(), device=torch.device('cpu'))
    coords = coords.type(torch.float) * tsdf.voxel_size + tsdf.origin.T
    mask = tsdf.tsdf_vol.abs().view(-1)<1

    # transfer vertex instance ids to voxels near surface
    instance_vol = torch.zeros(len(mask), dtype=torch.long)
    for i in mask.nonzero():
        _, inds, dist = kdtree.search_knn_vector_3d(coords[:,i], 1)
        if dist[0]<dist_thresh:
            instance_vol[i] = instance_verts[inds[0]]

    tsdf.attribute_vols['instance'] = instance_vol.view(list(tsdf.tsdf_vol.size()))
    tsdf.save(data['file_name_vol_%02d'%voxel_size])

    key = 'vol_%02d'%voxel_size
    temp_data = {key:tsdf, 'instances':data['instances'], 'dataset':data['dataset']}
    tsdf = transforms.InstanceToSemseg('nyu40')(temp_data)[key]
    mesh = tsdf.get_mesh('semseg')
    fname = data['file_name_vol_%02d'%voxel_size]
    mesh.export(fname.replace('tsdf', 'mesh').replace('.npz','_semseg.ply'))


def prepare_scannet(path, path_meta, i=0, n=1, test_only=False, max_depth=3):
    """ Create all derived data need for the Scannet dataset

    For each scene an info.json file is created containg all meta data required
    by the dataloaders. We also create the ground truth TSDFs by fusing the
    ground truth TSDFs and add semantic labels

    Args:
        path: path to the scannet dataset
        path_meta: path to save all the derived data
            (we recommend creating a parallel directory structure so that 
            we don't modify the original dataset)
        i: process id (used for parallel processing)
            (this process operates on scenes [i::n])
        n: number of processes
        test_only: only prepare the test set (for rapid testing if you dont 
            plan to train)
        max_depth: mask out large depth values since they are noisy

    Returns:
        Writes files to path_meta
    """
    
    scenes = []
    if not test_only:
        scenes += sorted([os.path.join('scans', scene) 
                          for scene in os.listdir(os.path.join(path, 'scans'))])
    scenes += sorted([os.path.join('scans_test', scene)
                      for scene in os.listdir(os.path.join(path, 'scans_test'))])

    scenes = scenes[i::n]

    if i==0:
        prepare_scannet_splits(path, path_meta)

    for scene in scenes:
        prepare_scannet_scene(scene, path, path_meta)
        for voxel_size in [4,8,16]:
            fuse_scene(path_meta, scene, voxel_size, device=i%8, max_depth=max_depth)
            if scene.split('/')[0]=='scans':
                label_scene(path_meta, scene, voxel_size)



# def prepare_rio(path, path_meta, i=0, n=1):
#     """ Create all derived data need for the RIO dataset

#     For each scene an info.json file is created containg all meta data required
#     by the dataloaders. We also create the ground truth TSDFs by fusing the
#     ground truth TSDFs and add semantic labels

#     Args:
#         path: path to the scannet dataset
#         path_meta: path to save all the derived data
#             (we recommend creating a parallel directory structure so that 
#             we don't modify the original dataset)
#         i: process id (used for parallel processing)
#             (this process operates on scenes [i::n])
#         n: number of processes
#         max_depth: mask out large depth values since they are noisy

#     Returns:
#         Writes files to path_meta
#     """
#     scenes = sorted([
#         scene
#         for scene in os.listdir(path)
#         if os.path.isdir(os.path.join(path,scene)) and 
#             os.path.exists(os.path.join(path, scene, 
#                 'sequence', 'frame-%06d.pose.txt'%0))
#     ])

#     scenes = scenes[i::n]

#     if i==0:
#         prepare_rio_splits(path, path_meta)

#     for scene in scenes:
#         prepare_rio_scene(scene, path, path_meta)
#         for voxel_size in [4,8,16]:
#             fuse_scene(path_meta, scene, voxel_size, device=i%8, max_depth=max_depth)
#             label_scene(path_meta, scene, voxel_size)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf on Scannet')
    parser.add_argument("--path", required=True, metavar="DIR",
        help="path to raw dataset")
    parser.add_argument("--path_meta", required=True, metavar="DIR",
        help="path to store processed (derived) dataset")
    parser.add_argument("--dataset", required=True, type=str,
        help="which dataset to prepare")
    parser.add_argument('--i', default=0, type=int,
        help='index of part for parallel processing')
    parser.add_argument('--n', default=1, type=int,
        help='number of parts to devide data into for parallel processing')
    parser.add_argument('--test', action='store_true',
        help='only prepare the test set (for rapid testing if you dont plan to train)')
    parser.add_argument('--max_depth', default=3., type=float,
        help='mask out large depth values since they are noisy')
    args = parser.parse_args()

    i=args.i
    n=args.n
    assert 0<=i and i<n

    if args.dataset == 'sample':
        scenes = ['sample1']
        scenes = scenes[i::n] # distribute among workers
        for scene in scenes:
            prepare_sample_scene(
                scene,
                os.path.join(args.path, 'sample'),
                os.path.join(args.path_meta, 'sample'),
            )

    elif args.dataset == 'scannet':
        prepare_scannet(
            os.path.join(args.path, 'scannet'),
            os.path.join(args.path_meta, 'scannet'),
            i,
            n,
            args.test,
            args.max_depth
        )

    # elif args.dataset == 'rio':
    #     prepare_rio(
    #         os.path.join(args.path, 'RIO'),
    #         os.path.join(args.path_meta, 'RIO'),
    #         i,
    #         n
    #     )

    else:
        raise NotImplementedError('unknown dataset %s'%args.dataset)