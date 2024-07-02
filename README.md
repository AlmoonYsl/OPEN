# OPEN
## [ECCV 2024] OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection

## News
- [2024-07.02] OPEN is accepted by ECCV 2024.

## Results on NuScenes Val Set.
| Model | Backbone | Resolution | NDS  | mAP  | Config | Ckpt |
|:-----:|:--------:|:----------:|:----:|:----:|:------:|:----:|
| OPEN  |   R50    | 256 x 704  | 56.4 | 46.5 |       |      |
| OPEN  |   R101   | 512 x 1408 | 60.6 | 51.6 |       |      |

## Results on NuScenes Test Set.
| Model | Backbone | Resolution | NDS  | mAP  |
|:-----:|:--------:|:----------:|:----:|:----:|
| OPEN  |  V2-99   | 640 x 1600 | 64.4 | 56.7 |

## TODO
- [ ] Release the paper.
- [ ] Release the code of OPEN.

## Acknowledgements
We thank these great works and open-source repositories:
[PETR](https://github.com/megvii-research/PETR), [StreamPETR](https://github.com/exiawsh/StreamPETR), and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).
