# 数字图像处理课程作业

- 课程名称：数字图像处理
- 授课老师：郭玉东
- 作者：郑登煜 PB22010397

本仓库用于整理《数字图像处理》课程的作业实现、实验结果与报告。

## 作业目录

- [作业一：Image Warping](Assignment_01_ImageWarping/README.md)
- [作业二：DIP with PyTorch](Assignment_02_DIPwithPyTorch/README.md)

## 作业二内容说明

[Assignment_02_DIPwithPyTorch](Assignment_02_DIPwithPyTorch/README.md) 包含两部分内容：

1. `Poisson Image Editing`
   使用 PyTorch 在梯度域中优化融合图像，通过交互式多边形选区将前景区域自然融合到背景图像中。

2. `Pix2Pix-style Image Translation`
   使用全卷积网络和 `maps` 数据集完成图像到图像映射任务，并保存训练过程中的可视化结果与模型权重。

作业二报告中已经给出了：

- 方法原理说明
- 代码实现要点
- 实验设置
- Poisson 三组实验结果
- Pix2Pix 的训练/验证结果分析
