# T2EA: Target-aware Taylor Expansion Approximation Network for Infrared and Visible Image Fusion
This is official Pytorch implementation of ["T2EA: Target-aware Taylor Expansion Approximation Network for Infrared and Visible Image Fusion"](https://ieeexplore.ieee.org/document/10819442).

## Architecture
![The overall framework of the Target-aware Taylor Expansion Approximation Network for Infrared and Visible Image Fusion.](https://github.com/MysterYxby/T2EA/blob/main/Figure/framework.jpg)

## Recommended Environment
- Python 3.7
- Pytorch >= 1.8
## Training Dataset
Please download the dataset [MSRS](https://github.com/Linfeng-Tang/MSRS).

## To train
### 1 To train the Taylor network
Run "train_TEM.py " (Plz. set the relevant parameters and dirs).

### 2 To train the Fusion network
Run "train_Fusion.py " (Plz. set the relevant parameters and dirs).

### 3 To train the Fusion network with Seg Network
Run "train_task.py " (Plz. set the relevant parameters and dirs (the dirs in "Datates.py ")).

## To test
Run "test_image.py " (Plz. set the relevant parameters and models' dirs).

## Acknowledgement
The dataset is sourced from Tang et al. And our code is partly constructed on SeAFusion.

[1] L. Tang, J. Yuan, H. Zhang, X. Jiang, and J. Ma, “PIAFusion: A progressive infrared and visible image fusion network based on illumination aware,” Inf. Fusion, vols. 83–84, pp. 79–92, Jul. 2022.

[2] L. Tang, J. Yuan, and J. Ma, “Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network,” Inf. Fusion, vol. 82, pp. 28–42, Jun. 2022.

## If this work is helpful to you, please cite it as：
```
@ARTICLE{Huang2025T2EA,
  author={Huang, Zhenghua and Lin, Cheng and Xu, Biyun and Xia, Menghan and Li, Qian and Li, Yansheng and Sang, Nong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={T2EA: Target-Aware Taylor Expansion Approximation Network for Infrared and Visible Image Fusion}, 
  year={2025},
  volume={35},
  number={5},
  pages={4831-4845},
  doi={10.1109/TCSVT.2024.3524794}}
```
### If you have any question about this code, feel free to reach me (cloudxu08@outlook.com)
