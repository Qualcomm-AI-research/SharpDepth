# 📦 Data Preparation Guide
We provide dataloaders for the following datasets, adapted from the Marigold codebase:

* KITTI (`data/kitti_dataset.py`)
* NYUv2 (`data/nyu_dataset.py`)
* Sintel (`data/sintel_dataset.py`)
* Spring (`data/spring_dataset.py`)
* UnrealStereo4K (`data/unrealstereo_dataset.py`)

All datasets should be placed under a common directory following this structure:

```
SharpDepth
├── datasets
│   ├── kitti
│   │   ├── 2011_09_26
│   │   ├── 2011_09_26_drive_0002_sync
│   │   ├── ...
│   ├── nyuv2
│   │   ├── train
│   │   ├── test
│   │   │   ├── bathroom_0003
│   │   │   │   |── rgb_0001.png
│   │   │   │   |── depth_0001.png
│   │   │   │   |── ...
│   ├── spring
│   │   ├── 0001
│   │   │   ├── frame_left
│   │   │   │   |── frame_left_0001.png
│   │   │   │   |── ...
│   │   │   ├── disp1_left 
│   │   │   │   |── disp1_left_0001.png
│   │   │   │   |── ...
│   │   ├── 0002
│   │   │   ├── frame_left
│   │   │   │   |── ...
│   │   │   ├── disp1_left      
│   │   │   │   |── ...
│   ├── sintel
│   │   ├── training
│   │   │   ├── final
│   │   │   │   |── frame_left_0001.png
│   │   │   │   |── ...
│   │   │   ├── depth
│   │   │   │   |── frame_left_0001.dpt
│   │   │   │   |── ...
│   ├── unreal
│   │   ├── 00008
│   │   │   ├── Image0
│   │   │   │   |── 000001.png
│   │   │   │   |── ...
│   │   │   ├── Disp0
│   │   │   │   |── 000001.npy
│   │   │   │   |── ...
```

All datasets should be placed under a common directory following this structure:

| Dataset Name       | Download Links                                                                 |
|--------------------|---------------------------------------------------------------------------------|
| KITTI              | [image & ground-truths](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/kitti/kitti_eigen_split_test.tar/) |
| NYU | [image & ground-truths](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/nyuv2/nyu_labeled_extracted.tar)                                                                   |
| Sintel  | [image & ground-truths](http://sintel.is.tue.mpg.de/downloads)                                                                    |
| UnrealStereo4K  | [image & ground-truths](https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_00008.zip)                                                                    |
| Spring  | [image](https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-3376/14&version=2.0) & [ground-truths](https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-3376/2&version=2.0)                                                                   |

