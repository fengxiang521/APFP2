# APFP
The code of APFP

# Introduction
In this repo, we provide the implementation of the following paper:
Adaptive Prototype Few-Shot Image Classification Method Based on Feature Pyramid

In this paper, a novel feature extraction method termed FResNet is introduced, which leverages feature pyramid structures to retain finer details during computation, resulting in feature maps with enhanced detailed features. Addressing the issue of utilizing sample mean for class prototypes in ProtoNet, we proposed a novel class prototype computation method called Adaptive Prototype. The Adaptive Prototype method adaptively computes optimal support set class prototypes based on the similarity between each support set sample and the query sample, yielding prototypes more aligned with the query sample features. Finally, the APFP method proposed in this paper was evaluated on the MiniImagenet and CUB datasets, demonstrating significant improvements compared to previous methods, achieving state-of-the-art performance on both datasets.

# Start
## Dataset
MiniImagenet: Download Link: [[BaiduCloud](https://pan.baidu.com/s/1Wi06keM-1WXP26YqwdpaFw?pwd=ankq)] [[GoogleDrive](https://drive.google.com/file/d/1aBxfcU5cn-htIlqriiOQCOXp_t9TOm9g/view?usp=sharing)].

CUB: Download Link: [[BaiduCloud](https://pan.baidu.com/s/1JyVQC1-cLiPIl6yYAdlkeA?pwd=yrv1)] [[GoogleDrive](https://drive.google.com/file/d/1sbOiZP-U4A7NdhkJo7YzeffNf5GatIwk/view?usp=sharing)].

Download the Mini-ImageNet dataset and the CUB dataset. Set the dataset paths in the `run_test.sh` script.

## Models
MiniImagenet:
[[5way1shot](https://pan.baidu.com/s/1E7W7upbyBejgIMkeT7HPjA?pwd=f0t9)]  [[5way5shot](https://pan.baidu.com/s/1rQINbaOMie2XzNKCenwmDA?pwd=3tom)]

CUB：
[[5way1shot](https://pan.baidu.com/s/15K2u6RX7rZFTJxccSJdDCQ?pwd=k1w1)]  [[5way5shot](https://pan.baidu.com/s/1k-cVTGwLZYVljiHSIRZZNw?pwd=wq0e)]


## Test

MiniImagenet test: 
```shell
cd script/mini-image/
./run_test.sh
```

cub test:
```shell
cd script/cub/
./run_test.sh
```
# Implementation environment
Note that the test accuracy may slightly vary with different Pytorch/CUDA versions, GPUs, etc.

Linux
Python 3.8.
torch: 1.11.0+cu113
GPU (RTX3090) + CUDA11.3.109

# Acknowledgments
Our code builds upon the the following code publicly available:[[DeepBDC](https://github.com/Fei-Long121/DeepBDC)]
