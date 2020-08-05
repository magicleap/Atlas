
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

import numpy as np
import pyrender
import torch
import trimesh

from atlas.data import SceneDataset, parse_splits_list
from atlas.evaluation import eval_tsdf, eval_mesh, eval_depth, project_to_mesh
import atlas.transforms as transforms
from atlas.tsdf import TSDF, TSDFFusion
from visualize_metrics import visualize


class Renderer():
    """OpenGL mesh renderer 
    
    Used to render depthmaps from a mesh for 2d evaluation
    """
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        #self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)#, self.render_flags) 

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R =  np.array([[1, 0, 0],
                       [0, c, -s],
                       [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose@axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
        


def process(info_file, save_path, total_scenes_index, total_scenes_count):

    # gt depth data loader
    width, height = 640, 480
    transform = transforms.Compose([
        transforms.ResizeImage((width,height)),
        transforms.ToTensor(),
    ])
    dataset = SceneDataset(info_file, transform, frame_types=['depth'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                             batch_sampler=None, num_workers=2)
    scene = dataset.info['scene']

    # get info about tsdf
    file_tsdf_pred = os.path.join(save_path, '%s.npz'%scene)
    temp = TSDF.load(file_tsdf_pred)
    voxel_size = int(temp.voxel_size*100)
    
    # re-fuse to remove hole filling since filled holes are penalized in 
    # mesh metrics
    vol_dim = list(temp.tsdf_vol.shape)
    origin = temp.origin
    tsdf_fusion = TSDFFusion(vol_dim, float(voxel_size)/100, origin, color=False)
    device = tsdf_fusion.device

    # mesh renderer
    renderer = Renderer()
    mesh_file = os.path.join(save_path, '%s.ply'%scene)
    mesh = trimesh.load(mesh_file, process=False)
    mesh_opengl = renderer.mesh_opengl(mesh)

    for i, d in enumerate(dataloader):
        if i%25==0:
            print(total_scenes_index, total_scenes_count,scene, i, len(dataloader))

        depth_trgt = d['depth'].numpy()
        _, depth_pred = renderer(height, width, d['intrinsics'], d['pose'], mesh_opengl)

        temp = eval_depth(depth_pred, depth_trgt)
        if i==0:
            metrics_depth = temp
        else:
            metrics_depth = {key:value+temp[key] 
                             for key, value in metrics_depth.items()}

        # # play video visualizations of depth
        # viz1 = (np.clip((depth_trgt-.5)/5,0,1)*255).astype(np.uint8)
        # viz2 = (np.clip((depth_pred-.5)/5,0,1)*255).astype(np.uint8)
        # viz1 = cv2.applyColorMap(viz1, cv2.COLORMAP_JET)
        # viz2 = cv2.applyColorMap(viz2, cv2.COLORMAP_JET)
        # viz1[depth_trgt==0]=0
        # viz2[depth_pred==0]=0
        # viz = np.hstack((viz1,viz2))
        # cv2.imshow('test', viz)
        # cv2.waitKey(1)

        tsdf_fusion.integrate((d['intrinsics'] @ d['pose'].inverse()[:3,:]).to(device),
                              torch.as_tensor(depth_pred).to(device))


    metrics_depth = {key:value/len(dataloader) 
                     for key, value in metrics_depth.items()}

    # save trimed mesh
    file_mesh_trim = os.path.join(save_path, '%s_trim.ply'%scene)
    tsdf_fusion.get_tsdf().get_mesh().export(file_mesh_trim)

    # eval tsdf
    file_tsdf_trgt = dataset.info['file_name_vol_%02d'%voxel_size]
    metrics_tsdf = eval_tsdf(file_tsdf_pred, file_tsdf_trgt)

    # eval trimed mesh
    file_mesh_trgt = dataset.info['file_name_mesh_gt']
    metrics_mesh = eval_mesh(file_mesh_trim, file_mesh_trgt)

    # transfer labels from pred mesh to gt mesh using nearest neighbors
    file_attributes = os.path.join(save_path, '%s_attributes.npz'%scene)
    if os.path.exists(file_attributes):
        mesh.vertex_attributes = np.load(file_attributes)
        print(mesh.vertex_attributes)
        mesh_trgt = trimesh.load(file_mesh_trgt, process=False)
        mesh_transfer = project_to_mesh(mesh, mesh_trgt, 'semseg')
        semseg = mesh_transfer.vertex_attributes['semseg']
        # save as txt for benchmark evaluation
        np.savetxt(os.path.join(save_path, '%s.txt'%scene), semseg, fmt='%d')
        mesh_transfer.export(os.path.join(save_path, '%s_transfer.ply'%scene))

        # TODO: semseg val evaluation

    metrics = {**metrics_depth, **metrics_mesh, **metrics_tsdf}
    print(metrics)

    rslt_file = os.path.join(save_path, '%s_metrics.json'%scene)
    json.dump(metrics, open(rslt_file, 'w'))

    return scene, metrics



def main():
    parser = argparse.ArgumentParser(description="Atlas Testing")
    parser.add_argument("--model", required=True, metavar="FILE",
                        help="path to checkpoint")
    parser.add_argument("--scenes", default="data/scannet_test.txt",
                        help="which scene(s) to run on")
    args = parser.parse_args()

    # get all the info_file.json's from the command line
    # .txt files contain a list of info_file.json's
    info_files = parse_splits_list(args.scenes)
    # info_files=[info_files[0]]

    metrics = {}
    for i, info_file in enumerate(info_files):
        # run model on each scene
        scene, temp = process(info_file, args.model, i, len(info_files))
        metrics[scene] = temp

    rslt_file = os.path.join(args.model, 'metrics.json')
    json.dump(metrics, open(rslt_file, 'w'))

    # display results
    visualize(rslt_file)

if __name__ == "__main__":
    main()



    # # zip up semseg results for benchmark submission
    # cmd = 'zip -j %s/semseg.zip %s/*.txt'%(save_path, save_path)
    # os.system(cmd)

    # # pretty print metrics
    # print()
    # metrics_keys = list(list(metrics.values())[0].keys())
    # print(''.join( [key.ljust(15) for key in ['scene']+metrics_keys] ))
    # for scene, metrics_i in metrics.items():
    #     metrics_i_fmt = ['%03.3f'%value for value in metrics_i.values()]
    #     print(''.join([s.ljust(15) for s in [scene]+metrics_i_fmt]))
    # metrics_avg = [np.mean([metrics[scene][key] for scene in metrics.keys()]) 
    #                for key in metrics_keys]
    # print()
    # metrics_avg_fmt = ['%03.3f'%value for value in metrics_avg]
    # print(''.join([s.ljust(15) for s in ['average']+metrics_avg_fmt]))