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
import json
import imageio
from skimage.transform import resize

from atlas.data import load_info_json, parse_splits_list
from atlas.evaluation import eval_mesh, eval_depth


def read_array(path):
    # from https://github.com/colmap/colmap
    # Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def eval_scene(info_file, pathout):
    """ Evaluates COLMAP inference compared to ground truth

    Args:
        info_file: path to info_json file for the scene
        pathout: path where intermediate and final results are stored
    """

    info = load_info_json(info_file)
    dataset = info['dataset']
    scene = info['scene']
    frames = info['frames']

    fnames = os.listdir(os.path.join(pathout, dataset, scene, 'stereo', 'depth_maps'))
    frames = [frame for frame in frames 
              if os.path.split(frame['file_name_image'])[1] + '.geometric.bin' in fnames]

    # 2d depth metrics
    for i, frame in enumerate(frames):
        if i%25 == 0:
            print(scene, i, len(fnames))

        
        fname_trgt = frame['file_name_depth']
        fname_pred = os.path.join(pathout, dataset, scene, 'stereo', 'depth_maps',
            os.path.split(frame['file_name_image'])[1]+'.geometric.bin')
        depth_trgt = imageio.imread(fname_trgt).astype('float32') / 1000
        depth_pred = read_array(fname_pred)
        depth_pred[depth_pred>5]=0 # ignore depth beyond 5 meters as it is probably wrong
        depth_pred = resize(depth_pred, depth_trgt.shape)

        temp = eval_depth(depth_pred, depth_trgt)
        if i==0:
            metrics_depth = temp
        else:
            metrics_depth = {key:value+temp[key] 
                             for key, value in metrics_depth.items()}
    metrics_depth = {key:value/len(frames) 
                     for key, value in metrics_depth.items()}

    # 3d point metrics
    fname_pred = os.path.join(pathout, dataset, scene, 'fused.ply')
    fname_trgt = info['file_name_mesh_gt']
    metrics_mesh = eval_mesh(fname_pred, fname_trgt)

    metrics = {**metrics_depth, **metrics_mesh}
    print(metrics)

    rslt_file = os.path.join(pathout, dataset, scene, 'metrics.json')
    json.dump(metrics, open(rslt_file, 'w'))

    return metrics

    



def main():
    parser = argparse.ArgumentParser(description='Evaluate COLMAP')
    parser.add_argument("--scenes", default="data/scannet_test.txt",
        help="path to raw dataset")
    parser.add_argument("--pathout", required=True, metavar="DIR",
        help="path to store processed (derived) dataset")
    args = parser.parse_args()

    scenes = parse_splits_list(args.scenes)

    metrics = {}
    for scene in scenes:
        metrics[scene] = eval_scene(scene, args.pathout)
    
    rslt_file = os.path.join(args.pathout, 'metrics.json')
    json.dump(metrics, open(rslt_file, 'w'))


if __name__ == "__main__":
    main()
