DeepBranch: Deep Neural Networks for Branch Point Detection in Biomedical Images
===
Here we provide the code and models for the following paper
DeepBranch: Deep Neural Networks for Branch Point Detection in Biomedical Images
Yinghui Tan, Min Liu, Weixun Chen, Xueping Wang, and Hanchuan Peng, Senior Member, IEEE

Updates
---
* April 17, 2019

Overview
---
1.	We propose to use an improved 3D U-Net model to segment candidate regions containing branch points in this detection tasks. It can produce dense outputs with the same size of inputs instead of classifying by pixel, which greatly decreased the computation times and memory requirements. Meanwhile a combination of global and local features extracted by using anisotropic convolution kernels from different levels can help to enhance the robustness to the images from different dates;
2.	We propose a novel multi-scale multi-view convolutional neural networks (MSM-Net) to separate the real branch points from false positives. The MSM-Net is constructed by combining various streams of CNNs. Each stream processes the sampled patches from a specific view, for which the outputs units are combined using feature fusion technology to get the final branch point results. In addition, the multi-scale schedule solved the problems coming with the variety sizes.
3.	we have trained and tested the two-stage cascade framework on neuron image stacks from BigNeuron Project [16], bronchus images and retinal blood vessel images respectively after we annotated numbers of branch point ground truths manually. Besides, the detected branch points have been used to improve and diagnose the existing reconstruction algorithms.







