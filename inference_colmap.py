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
from PIL import Image
from scipy.spatial.transform import Rotation as R
import sqlite3

from atlas.data import load_info_json, parse_splits_list

def process(info_file, pathout, stride, scale):
    """ Run Colmap dense reconstruction with ground truth pose.

    Copies and creates the necessary file structure required by Colmap.
    Then runs Colmap.

    Args:
        info_file: path to info_json file for the scene
        pathout: path to store intermediate and final results
        stride: number of frames to skip (reduces runtime)
        scale: how much to downsample images (reduces runtime and often 
            improves stereo matching results)
    """

    info = load_info_json(info_file)
    dataset = info['dataset']
    scene = info['scene']
    frames = info['frames'][::stride]

    os.makedirs(os.path.join(pathout, dataset, scene, 'images'), exist_ok=True)

    for i, frame in enumerate(frames):
        if i%25 == 0:
            print(i,len(frames))

        img = Image.open(frame['file_name_image'])
        w = img.width//scale
        h = img.height//scale
        fname_out = os.path.split(frame['file_name_image'])[1]
        fname_out = os.path.join(pathout, dataset, scene, 'images', fname_out)
        img.resize((w,h), Image.BILINEAR).save(fname_out)

    with open(os.path.join(pathout, dataset, scene, 'cameras.txt'), 'w') as fp:
        fp.write('1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}'.format(
            w=w,
            h=h,
            fx=frames[0]['intrinsics'][0][0]/scale,
            fy=frames[0]['intrinsics'][1][1]/scale,
            cx=frames[0]['intrinsics'][0][2]/scale,
            cy=frames[0]['intrinsics'][1][2]/scale,
        ))
    
    with open(os.path.join(pathout, dataset, scene, 'points3D.txt'), 'w') as fp:
        pass

   
    cmd = 'colmap feature_extractor --database_path %s --image_path %s'%(
        os.path.join(pathout, dataset, scene, 'database.db'),
        os.path.join(pathout, dataset, scene, 'images')
    )
    os.system(cmd)
    cmd = 'colmap exhaustive_matcher --database_path %s'%(
        os.path.join(pathout, dataset, scene, 'database.db')
    )
    os.system(cmd)


    
    conn = sqlite3.connect(os.path.join(pathout, dataset, scene, 'database.db'))
    c = conn.cursor()
    c.execute('SELECT image_id, name FROM images')
    db_list = sorted(c.fetchall(), key=lambda x:x[1])
    pose_dict = {os.path.split(frame['file_name_image'])[1]: np.array(frame['pose'])
                 for frame in frames}
    with open(os.path.join(pathout, dataset, scene, 'images.txt'), 'w') as fp:
        for ind, name in db_list:

            pose = pose_dict[name]
            pose = np.linalg.inv(pose)
            q = R.from_matrix(pose[:3,:3]).as_quat()
            t = pose[:3,3]

            fp.write('{i}, {qw}, {qx}, {qy}, {qz}, {tx}, {ty}, {tz}, 1, {name}\n\n'.format(
                i=ind,
                qw=q[3],
                qx=q[0],
                qy=q[1],
                qz=q[2],
                tx=t[0],
                ty=t[1],
                tz=t[2],
                name=name
            ))

    cmd = ('colmap point_triangulator --database_path %s --image_path %s'
           ' --input_path %s --output_path %s')%(
        os.path.join(pathout, dataset, scene, 'database.db'),
        os.path.join(pathout, dataset, scene, 'images'),
        os.path.join(pathout, dataset, scene),
        os.path.join(pathout, dataset, scene)
    )
    os.system(cmd)

    cmd = 'colmap image_undistorter --image_path %s --input_path %s --output_path %s'%(
        os.path.join(pathout, dataset, scene, 'images'),
        os.path.join(pathout, dataset, scene),
        os.path.join(pathout, dataset, scene)
    )
    os.system(cmd)

    cmd = 'colmap patch_match_stereo --workspace_path %s'%(
        os.path.join(pathout, dataset, scene)
    )
    os.system(cmd)

    cmd = 'colmap stereo_fusion --workspace_path %s --output_path %s'%(
        os.path.join(pathout, dataset, scene),
        os.path.join(pathout, dataset, scene, 'fused.ply')
    )
    os.system(cmd)

    
    cmd = ('colmap delaunay_mesher --input_path %s --output_path %s'
           ' --DelaunayMeshing.quality_regularization 5.'
           ' --DelaunayMeshing.max_proj_dist 10')%(
        os.path.join(pathout, dataset, scene),
        os.path.join(pathout, dataset, scene+'.ply')
    )
    os.system(cmd)



def main():
    parser = argparse.ArgumentParser(description='Inference with COLMAP')
    parser.add_argument("--scenes", default="data/scannet_test.txt",
        help="path to raw dataset")
    parser.add_argument("--pathout", required=True, metavar="DIR",
        help="path to store processed (derived) dataset")
    parser.add_argument('--stride', default=2, type=int,
        help='number of frames to skip (imroves runtime)')
    parser.add_argument('--scale', default=4, type=int,
        help='factor to downsample images by (imroves runtime and quality)')
    parser.add_argument('--i', default=0, type=int,
        help='index of part for parallel processing')
    parser.add_argument('--n', default=1, type=int,
        help='number of parts to devide data into for parallel processing')
    args = parser.parse_args()

    i=args.i
    n=args.n
    assert 0<=i and i<n

    scenes = parse_splits_list(args.scenes)

    scenes = scenes[i::n]
    for scene in scenes:
        process(scene, args.pathout, args.stride, args.scale)

main()