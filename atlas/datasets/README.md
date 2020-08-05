# Data Common Format
Our dataset reads all matadata from a common json format.
Each scene has a json which contains all the metadata required by the dataset.
This includes a list of image files, a list of depth files, a list of poses, etc.

We provide a function to parse Scannet scenes into our format.
Note that this does not duplicate or modify any of the original data,
it just stores the required metadata (including paths) in a info.json file.
To use with another dataset you need to generate these json files,
and then you can use our dataset.

## JSON format
```
{'dataset': 'scannet',
 'path': path,
 'scene': scene,
 'file_name_mesh_gt': '',
 'instances': {}, # dict mapping instance id to label id
 'frames': [{'file_name_image': '',
             'file_name_depth': '',
             'file_name_instance': '',
             'intrinsics': intrinsics,
             'pose': pose,
             }
             ...
           ]
 'file_name_seg_indices': '',  # optional (used for labeling the tsdf from scannet labels)
 'file_name_seg_groups': '',  # optional (used for labeling the tsdf from scannet labels)
}
```
