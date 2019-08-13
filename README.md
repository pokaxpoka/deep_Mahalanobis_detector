# A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks

This project is for the paper "[A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888)". Some codes are from [odin-pytorch](https://github.com/facebookresearch/odin), [LID](https://github.com/xingjunm/lid_adversarial_subspace_detection), and [adversarial_image_defenses](https://github.com/facebookresearch/adversarial_image_defenses).

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requries Pytorch package to be installed:

* [Pytorch](http://pytorch.org/): Only GPU version is available.
* [scipy](https://github.com/scipy/scipy)
* [scikit-learn](http://scikit-learn.org/stable/)

## Downloading Out-of-Distribtion Datasets
We use download links of two out-of-distributin datasets from [odin-pytorch](https://github.com/facebookresearch/odin):

* [Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* [LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

Please place them to `./data/`.

## Downloading Pre-trained Models
We provide six pre-trained neural networks (1) three DenseNets trained on CIFAR-10, CIFAR-100 and SVHN, where models trained on CIFAR-10 and CIFAR-100 are from [odin-pytorch](https://github.com/ShiyuLiang/odin-pytroch), and (2) three ResNets trained on CIFAR-10, CIFAR-100 and SVHN.

* [DenseNet on CIFAR-10](https://www.dropbox.com/s/pnbvr16gnpyr1zg/densenet_cifar10.pth?dl=0) / [DenseNet on CIFAR-100](https://www.dropbox.com/s/7ur9qo81u30od36/densenet_cifar100.pth?dl=0) / [DenseNet on SVHN](https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth?dl=0)
* [ResNet on CIFAR-10](https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth?dl=0) / [ResNet on CIFAR-100](https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth?dl=0) / [ResNet on SVHN](https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth?dl=0)

Please place them to `./pre_trained/`.

## Detecting Out-of-Distribution Samples (Baseline and ODIN)

```
# model: ResNet, in-distribution: CIFAR-10, gpu: 0
python OOD_Baseline_and_ODIN.py --dataset cifar10 --net_type resnet --gpu 0
```

## Detecting Out-of-Distribution Samples (Mahalanobis detector)

### 1. Extract detection characteristics:
```
# model: ResNet, in-distribution: CIFAR-10, gpu: 0
python OOD_Generate_Mahalanobis.py --dataset cifar10 --net_type resnet --gpu 0
```

### 2. Train simple detectors:
```
# model: ResNet
python OOD_Regression_Mahalanobis.py --net_type resnet
```

## Detecting Adversarial Samples (LID & Mahalanobis detector)

### 0. Generate adversarial samples:
```
# model: ResNet, in-distribution: CIFAR-10, adversarial attack: FGSM  gpu: 0
python ADV_Samples.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0
```

### 1. Extract detection characteristics:
```
# model: ResNet, in-distribution: CIFAR-10, adversarial attack: FGSM  gpu: 0
python ADV_Generate_LID_Mahalanobis.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0
```

### 2. Train simple detectors:
```
# model: ResNet
python ADV_Regression.py --net_type resnet
```
