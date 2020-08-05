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

import csv
import json
import os

import numpy as np


def load_rio_label_mapping(path):
    mapping = {}
    with open(os.path.join(path, "mapping.txt")) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            id, name = int(line[0]), line[1]
            mapping[name] = id
    return mapping

def load_rio_nyu40_mapping():
    mapping = {0:0}
    if os.path.exists('data/rio_mapping.txt'):
        with open('data/rio_mapping.txt') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for i, line in enumerate(tsvreader):
                if i==0:
                    continue
                id, nyu40id = int(line[0]), int(line[2])
                mapping[id] = nyu40id
    return mapping


def parse_intrinsics(fname):
    data = dict([line.rstrip().split(' = ') for line in open(fname, 'r')])
    intrinsics_color = np.array([
        float(x) for x in data['m_calibrationColorIntrinsic'].split()
    ]).reshape(4,4)[:3,:3]
    intrinsics_depth = np.array([
        float(x) for x in data['m_calibrationDepthIntrinsic'].split()
    ]).reshape(4,4)[:3,:3]

    assert data['m_colorWidth']=='960'
    assert data['m_colorHeight']=='540'
    assert data['m_depthWidth']=='224'
    assert data['m_depthHeight']=='172'
    assert data['m_depthShift']=='1000'
    assert data['m_calibrationColorExtrinsic']=='1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'
    assert data['m_calibrationDepthExtrinsic']=='1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'

    assert intrinsics_depth[0,0]/224*960 - intrinsics_color[0,0] < 1
    assert intrinsics_depth[0,2]/224*960 - intrinsics_color[0,2] < 1
    assert intrinsics_depth[1,1]/172*540 - intrinsics_color[1,1] < 1
    assert intrinsics_depth[1,2]/172*540 - intrinsics_color[1,2] < 1

    return intrinsics_color

    #print(data)

def prepare_rio_scene(scene, path, path_meta, verbose=2):
    if verbose>0:
        print('preparing %s'%scene)

    data = {'dataset': 'rio',
            'path': path,
            'scene': scene,
            'file_name_mesh_gt': os.path.join(path, scene, 'mesh.refined.obj'),
            'file_name_seg_indices': os.path.join(
                path, scene, 'mesh.refined.0.010000.segs.json'),
            'file_name_seg_groups': os.path.join(
                path, scene, 'semseg.json')
           }

    # get instance id to category id
    label_mapping = load_rio_label_mapping(path)
    segGroups = json.load(open(
        os.path.join(path, scene, 'semseg.json'), 'r'
    ))['segGroups']
    data['instances'] = {}
    for seg in segGroups:
        name = seg['label']
        # "show case" is "showcase" in '41385827-a238-2435-8320-d8d3eb507f5e'.
        # also "bean bag" and "beanbag" are seperate classes in mapping.txt
        if name not in label_mapping:
            name = name.replace(' ', '')
        # some labels are not in mapping... map to unknown
        if name not in label_mapping:
            name = 'remove'
        data['instances'][seg['id']+1] = label_mapping[name]


    intrinsics = parse_intrinsics(os.path.join(path, scene, '_info.txt'))

    data['frames'] = []
    frame_ct = len(os.listdir(os.path.join(path, scene, 'sequence')))//3
    for i in range(frame_ct):
        if verbose>1 and i%25==0:
            print('preparing %s frame %d/%d'%(scene, i, frame_ct))

        pose = np.loadtxt(os.path.join(
            path, scene, 'sequence', 'frame-%06d.pose.txt'%i))

        # skip frames with no valid pose
        if not np.all(np.isfinite(pose)):
            continue

        file_name_image = os.path.join(
            path, scene, 'sequence', 'frame-%06d.color.jpg'%i)
        file_name_depth = os.path.join(
            path, scene, 'sequence', 'frame-%06d.depth.pgm'%i)
        file_name_instance = '' # TODO: code to render instance images from mesh

        frame = {"file_name_image":file_name_image,
                 "file_name_depth":file_name_depth,
                 "file_name_instance": file_name_instance, 
                 "intrinsics":intrinsics.tolist(),
                 "pose":pose.tolist(),
                }

        data['frames'].append(frame)

    os.makedirs(os.path.join(path_meta, scene),  exist_ok=True)
    json.dump(data, open(os.path.join(path_meta, scene, 'info.json'), 'w'))


def prepare_rio_splits(path, path_meta):
    """ Generates txt files for each dataset split

    Creates a folder 'data' in the current working directory.
    In that folder we create a split.txt file for each split containing
    the full path to the info.json for each scene in the split (one per line).
    Currently only creates a single split with all scenes that have GT (This is
    used for supplemental training data for the scannet benchmark).

    Args:
        path: Path to the original scannet data. This is used to get their
            standard splits.
        path_meta: Path to generated files.
    """

    os.makedirs('data', exist_ok=True)
    with open(os.path.join('data', 'rio_all.txt'), 'w') as out_file:
        scenes = [d for d in os.listdir(path) 
                  if os.path.isdir(os.path.join(path, d))]

        for scene in scenes:
            json_file = os.path.join(path_meta, scene, 'info.json')
            out_file.write(json_file+'\n')
            print(json_file)

    # copy mapping file into dataset splits for easy access
    os.system('cp %s/mapping.txt data/rio_mapping.txt'%path)
