# 数字图像处理课程作业

- 课程名称：数字图像处理
- 授课老师：郭玉东
- 作者：郑登煜 PB22010397

本仓库用于整理《数字图像处理》课程的作业实现、实验结果与报告。

## 作业目录

- [作业一：Image Warping](Assignment_01_ImageWarping/README.md)
- [作业二：DIP with PyTorch](Assignment_02_DIPwithPyTorch/README.md)
- [作业三：Bundle Adjustment](Assignment_03_BundleAdjustment/README.md)

## 作业一内容说明

[Assignment_01_ImageWarping](Assignment_01_ImageWarping/README.md) 实现了图像变形与映射相关内容，包含图像 warping 方法、实验结果和作业报告。

## 作业二内容说明

[Assignment_02_DIPwithPyTorch](Assignment_02_DIPwithPyTorch/README.md) 包含两部分内容：

1. `Poisson Image Editing`
   使用 PyTorch 在梯度域中优化融合图像，通过交互式多边形选区将前景区域自然融合到背景图像中。
2. `Pix2Pix-style Image Translation`
   使用全卷积网络和 `maps` 数据集完成图像到图像映射任务，并保存训练过程中的可视化结果与模型权重。

作业二报告中已经给出了方法原理说明、代码实现要点、实验设置、Poisson 三组实验结果，以及 Pix2Pix 的训练和验证结果分析。

## 作业三内容说明

[Assignment_03_BundleAdjustment](Assignment_03_BundleAdjustment/README.md) 包含两部分内容：

1. `Bundle Adjustment with PyTorch`
   从 50 个视角下的 2D 观测出发，使用 PyTorch 优化共享焦距、每个视角的相机外参以及 20000 个 3D 点坐标，并导出彩色点云和重投影可视化结果。
2. `3D Reconstruction with COLMAP`
   使用 COLMAP 对多视角图像进行特征提取、特征匹配、稀疏重建和 dense workspace 构建，保存相机模型、图像位姿和三维点云结果。

作业三报告中给出了 Bundle Adjustment 的投影模型、初始化策略、优化结果、loss 曲线、点云可视化、重投影结果，以及 COLMAP 三维重建流程和输出文件说明。
