<div align="center">

# OPEN

### [OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](https://arxiv.org/abs/2407.10753)

[Jinghua Hou](https://github.com/AlmoonYsl) <sup>1</sup>,
[Tong Wang](https://scholar.google.com/citations?user=EpUu4zIAAAAJ) <sup>2</sup>,
[Xiaoqing Ye](https://shuluoshu.github.io/)  <sup>2</sup>,
[Zhe Liu](https://github.com/happinesslz) <sup>1</sup>,
Shi Gong <sup>2</sup>,
[Xiao Tan](https://scholar.google.com/citations?user=R1rVRUkAAAAJ) <sup>2</sup>,<br>
[Errui Ding](https://scholar.google.com/citations?user=1wzEtxcAAAAJ) <sup>2</sup>,
[Jingdong Wang](https://jingdongwang2017.github.io/) <sup>2</sup>,
[Xiang Bai](https://xbai.vlrlab.net/) <sup>1,✉</sup>
<br>
<sup>1</sup> Huazhong University of Science and Technology,
<sup>2</sup> Baidu Inc.
<br>
✉ Corresponding author.
<br>

**ECCV 2024**

[![arXiv](https://img.shields.io/badge/arXiv-2407.10753-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2407.10753)

</div>

**Abstract** Accurate depth information is crucial for enhancing the performance of multi-view 3D object detection.
Despite the success of some existing multi-view 3D detectors utilizing pixel-wise depth supervision, they overlook two
significant phenomena: 1) the depth supervision obtained from LiDAR points is usually distributed on the surface of the
object,
which is not so friendly to existing DETR-based 3D detectors due to the lack of the depth of 3D object center; 2) for
distant objects, fine-grained depth estimation of the whole object is more challenging. Therefore, we argue that the
object-wise depth (or 3D center of the object) is essential for accurate detection. In this paper, we propose a new
multi-view 3D object detector named OPEN, whose main idea is to effectively inject object-wise depth information into
the network through our proposed object-wise position embedding. Specifically, we first employ an object-wise depth
encoder, which takes the pixel-wise depth map as a prior, to accurately estimate the object-wise depth. Then, we utilize
the proposed object-wise position embedding to encode the object-wise depth information into the transformer decoder,
thereby producing 3D object-aware features for final detection. Extensive experiments verify the effectiveness of our
proposed method. Furthermore, OPEN achieves a new state-of-the-art performance with 64.4% NDS and 56.7% mAP on the
nuScenes test benchmark.

![arch](assets/arch.jpg)

## News

* **2024.07.02**: Our another work [SEED](https://github.com/happinesslz/SEED) has also been accepted at ECCV 2024. 🎉
* **2024.07.02**: OPEN has been accepted at ECCV 2024. 🎉

## Results

* **nuScenes Val Set**

  The reproduced results are slightly higher than the reported results in the paper.
  R50：56.4 -> 56.5 NDS, 46.5 -> 47.0mAP

  R101: 60.6 -> 60.6 NDS, 51.6 -> 51.9 mAP

| Model | Backbone |                                                                                                  Pretrain                                                                                                  | Resolution | NDS  | mAP  |                      Config                      |                                           Download                                           |
|:-----:|:--------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------:|:----:|:----:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
| OPEN  |  V2-99   |                                                         [DD3D](https://drive.google.com/file/d/1a0qlGUUIOT1aqF-1iE9l181jkZfF2Hyf/view?usp=sharing)                                                         | 320 x 800  | 61.3 | 52.1 |  [config](projects/configs/open_vov_800_24e.py)  | [model](https://drive.google.com/file/d/1RgManSe09WPlucnnRwUngGSsSJMPHZkJ/view?usp=sharing)  |
| OPEN  |   R50    | [nuImage](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) | 256 x 704  | 56.5 | 47.0 |  [config](projects/configs/open_r50_704_90e.py)  | [model](https://drive.google.com/file/d/16L0NspLbZ53kaqNw3u29K9iwQ5gQVvh6/view?usp=sharing)  |
| OPEN  |   R101   |          [nuImage](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth)          | 512 x 1408 | 60.6 | 51.9 | [config](projects/configs/open_r101_1408_90e.py) | [model](https://drive.google.com/file/d/1X9S8TqPKc6522ckjpfQdyqk8qrSIs-Rp/view?usp=sharing)  |

* **nuScenes Test Set**

| Model | Backbone | Pretrain  | Resolution |   NDS    | mAP  |                          Config                          |                                           Download                                          |
|:-----:|:--------:|:---------:|:----------:|:--------:|:----:|:--------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
| OPEN  |  V2-99   | [DD3D]()  | 640 x 1600 |   64.4   | 56.7 | [config](projects/configs/open_vov_1600_60e_trainval.py) | [model](https://drive.google.com/file/d/1wa3CE0_9zy_UJk2kyeRJJAOg1nTaRJFd/view?usp=sharing) |

## TODO

- [x] Release the paper.
- [x] Release the code of OPEN.

## Citation

```
@inproceedings{
  hou2024open,
  title={OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection},
  author={Hou, Jinghua and Wang, Tong and Ye, Xiaoqing and Liu, Zhe and Tan, Xiao and Ding, Errui and Wang, Jingdong and Bai, Xiang},
  booktitle={ECCV},
  year={2024},
}
```

## Acknowledgements

We thank these great works and open-source repositories:
[PETR](https://github.com/megvii-research/PETR), [StreamPETR](https://github.com/exiawsh/StreamPETR),
and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).
