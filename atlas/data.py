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
import json

import numpy as np
from PIL import Image
import torch
import trimesh

from atlas.tsdf import TSDF
import atlas.transforms

DEPTH_SHIFT = 1000

def load_info_json(json_file):
    """ Open a json info_file and do a bit of preprocessing"""

    info = json.load(open(json_file,'r'))
    if 'instances' in info and info['instances'] is not None:
        # json doesn't store keys as ints so we cast here
        info['instances'] = {int(k):v for k,v in info['instances'].items()}
    else:
        info['instances'] = None
    return info
    

def map_frame(frame, frame_types=[]):
    """ Load images and metadata for a single frame.

    Given an info json we use this to load the images, etc for a single frame

    Args:
        frame: dict with metadata and paths to image files
            (see datasets/README)
        frame_types: which images to load (ex: depth, semseg, etc)

    Returns:
        dict containg metadata plus the loaded image
    """

    data = {key:value for key, value in frame.items()}
    data['image'] = Image.open(frame['file_name_image'])
    data['intrinsics'] = np.array(frame['intrinsics'], dtype=np.float32)
    data['pose'] = np.array(frame['pose'], dtype=np.float32)
    if 'depth' in frame_types:
        depth = Image.open(frame['file_name_depth'])
        depth = np.array(depth, dtype=np.float32) / DEPTH_SHIFT
        data['depth'] = Image.fromarray(depth)
    if 'semseg' in frame_types:
        if frame['file_name_instance']=='':
            data['instance'] = None
        else:
            data['instance'] = Image.open(frame['file_name_instance'])
    return data

def map_tsdf(info, data, voxel_types, voxel_sizes):
    """ Load TSDFs from paths in info.

    Args:
        info: dict with paths to TSDF files (see datasets/README)
        data: dict to add TSDF data to
        voxel_types: list of voxel attributes to load with the TSDF
        voxel_sizes: list of voxel sizes to load

    Returns:
        dict with TSDFs included
    """

    if len(voxel_types)>0:
        for scale in voxel_sizes:
            data['vol_%02d'%scale] = TSDF.load(info['file_name_vol_%02d'%scale],
                                               voxel_types)
    return data



class SceneDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, info_file, transform=None, frame_types=[],
                 voxel_types=[], voxel_sizes=[], num_frames=-1):
        """
        Args:
            info_file: path to json file (format described in datasets/README)
            transform: transform object to preprocess data
            frame_types: which images to load (ex: depth, semseg, etc)
            voxel_types: list of voxel attributes to load with the TSDF
            voxel_sizes: list of voxel sizes to load
            num_frames: number of evenly spaced frames to use (-1 for all)
        """

        self.info = load_info_json(info_file)
        self.transform = transform
        self.frame_types = frame_types
        self.voxel_types = voxel_types
        self.voxel_sizes = voxel_sizes

        # select evenly spaced subset of frames
        if num_frames>-1:
            length = len(self.info['frames'])
            inds = np.linspace(0, length-1, num_frames, dtype=int)
            self.info['frames'] = [self.info['frames'][i] for i in inds]


    def __len__(self):
        return len(self.info['frames'])

    def __getitem__(self, i):
        """
        Returns:
            dict of meta data and images for a single frame
        """

        frame = map_frame(self.info['frames'][i], self.frame_types)

        # put data in common format so we can apply transforms
        data = {'dataset': self.info['dataset'],
                'instances': self.info['instances'],
                'frames': [frame]}
        if self.transform is not None:
            data = self.transform(data)
        # remove data from common format and return the single frame
        data = data['frames'][0]

        return data

    def get_tsdf(self):
        """
        Returns:
            dict with TSDFs
        """

        # put data in common format so we can apply transforms
        data = {'dataset': self.info['dataset'],
                'instances': self.info['instances'],
                'frames': [],
               }

        # load tsdf volumes
        data = map_tsdf(self.info, data, self.voxel_types, self.voxel_sizes)

        # apply transforms
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_mesh(self):
        # TODO: also get vertex instances/semantics
        return trimesh.load(self.info['file_name_mesh_gt'], process=False)


class ScenesDataset(torch.utils.data.Dataset):
    """ Pytorch Dataset for a multiple scenes
    
    getitem loads a sequence of frames from a scene
    along with the corresponding TSDF for the scene
    """

    def __init__(self, info_files, num_frames, transform=None, frame_types=[],
                 frame_selection='random', voxel_types=[], voxel_sizes=[]):
        """
        Args:
            info_files: list of info_json files
            num_frames: number of frames in the sequence to load
            transform: apply preprocessing transform to images and TSDF
            frame_types: which images to load (ex: depth, semseg, etc)
            frame_selection: how to choose the frames in the sequence
            voxel_types: list of voxel attributes to load with the TSDF
            voxel_sizes: list of voxel sizes to load
        """

        self.info_files = info_files
        self.num_frames = num_frames
        self.transform = transform
        self.frame_types = frame_types
        self.frame_selection = frame_selection
        self.voxel_types = voxel_types
        self.voxel_sizes = voxel_sizes

    def __len__(self):
        return len(self.info_files)

    def __getitem__(self, i):
        """ Load images and TSDF for scene i"""

        info = load_info_json(self.info_files[i])

        frame_ids = self.get_frame_ids(info)
        # print(frame_ids)
        frames = [map_frame(info['frames'][i], self.frame_types)
                  for i in frame_ids]

        data = {'dataset': info['dataset'],
                'scene': info['scene'],
                'instances': info['instances'],
                'frames': frames}

        # load tsdf volumes
        data = map_tsdf(info, data, self.voxel_types, self.voxel_sizes)

        # apply transforms
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_frame_ids(self, info):
        """ Get the ids of the frames to load"""

        if self.frame_selection=='random':
            # select num_frames random frames from the scene
            return torch.randint(len(info['frames']), size=[self.num_frames])
        else:
            raise NotImplementedError('frame selection %s'%self.frame_selection)


def collate_fn(data_list):
    """ Flatten a set of items from ScenesDataset into a batch.

    Pytorch dataloader has memory issues with nested and complex 
    data structures. This flattens the data into a dict of batched tensors.
    Frames are batched temporally as well (bxtxcxhxw)
    """

    keys = list(data_list[0].keys())
    if len(data_list[0]['frames'])>0:
        frame_keys = list(data_list[0]['frames'][0].keys()) 
    else:
        frame_keys = []
    keys.remove('frames')

    out = {key:[] for key in keys+frame_keys}
    for data in data_list:
        for key in keys:
            out[key].append(data[key])

        for key in frame_keys:
            if torch.is_tensor(data['frames'][0][key]):
                out[key].append( torch.stack([frame[key] 
                                              for frame in data['frames']]) )
            else:
                # non tensor metadata may not exist for every frame
                # (ex: instance_file_name)
                out[key].append( [frame[key] if key in frame else None 
                                  for frame in data['frames']] )

    for key in out.keys():
        if torch.is_tensor(out[key][0]):
            out[key] = torch.stack(out[key])

    return out


def parse_splits_list(splits):
    """ Returns a list of info_file paths
    Args:
        splits (list of strings): each item is a path to a .json file 
            or a path to a .txt file containing a list of paths to .json's.
    """

    if isinstance(splits, str):
        splits = splits.split()
    info_files = []
    for split in splits:
        ext = os.path.splitext(split)[1]
        if ext=='.json':
            info_files.append(split)
        elif ext=='.txt':
            info_files += [info_file.rstrip() for info_file in open(split, 'r')]
        else:
            raise NotImplementedError('%s not a valid info_file type'%split)
    return info_files


