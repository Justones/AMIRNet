# All-in-one Multi-degradation Image Restoration Network via Hierarchical Degradation Representation (ACM MM23)

> **Authors** : [Cheng Zhang](https://scholar.google.com.hk/citations?user=L_Av9NcAAAAJ&hl=zh-CN), Yu Zhu, Qingsen Yan, Jinqiu Sun, Yanning Zhang
 
>  **Abstract** : The aim of image restoration is to recover high-quality images from distorted ones. However, current methods usually focus on a single task (\emph{e.g.}, denoising, deblurring or super-resolution) which cannot address the needs of real-world multi-task processing, especially on mobile devices. Thus, developing an all-in-one method that can restore images from various unknown distortions is a significant challenge. Previous works have employed contrastive learning to learn the degradation representation from observed images, but this often leads to representation drift caused by deficient positive and negative pairs. To address this issue, we propose a novel All-in-one Multi-degradation Image Restoration Network (AMIRNet) that can effectively capture and utilize accurate degradation representation for image restoration. AMIRNet learns a degradation representation for unknown degraded images by progressively constructing a tree structure through clustering, without any prior knowledge of degradation information. This tree-structured representation explicitly reflects the consistency and discrepancy of various distortions, providing a specific clue for image restoration. To further enhance the performance of the image restoration network and overcome domain gaps caused by unknown distortions, we design a feature transform block (FTB) that aligns domains and refines features with the guidance of the degradation representation. We conduct extensive experiments on multiple distorted datasets, demonstrating the effectiveness of our method and its advantages over state-of-the-art restoration methods both qualitatively and quantitatively.


### Dataset
The dataset used in our experiments consists of multi-degraded images sampled from SIDD, DDPD, LOL, and REDS. It can be downloaded from the [BaiduYun](https://pan.baidu.com/s/1tjNITPIeTMOKHfShNvxFgA?pwd=bcbf)[code:bcbf]. Please refer to our paper for more details.

### Dependencies & Installation
- NVIDIA GPU
- Anaconda

create a conda environment, run the command
```
conda create --name <env> --file requirements.txt
```
Then activate it

```
conda activate <env>
```

- - - - -
### Train & Test

The instructions on the training and test will be completed when we release the code.
- - - - -
### Results
We also provide the results of our method, please download from [here](https://pan.baidu.com/s/14nlMackn7B40FjoSds1voA?pwd=t5jj)[code: t5jj]
- - - - 
### Citation

```
@article{zhang2023all,
  title={All-in-one Multi-degradation Image Restoration Network via Hierarchical Degradation Representation},
  author={Zhang, Cheng and Zhu, Yu and Yan, Qingsen and Sun, Jinqiu and Zhang, Yanning},
  journal={arXiv preprint arXiv:2308.03021},
  year={2023}
}
```
