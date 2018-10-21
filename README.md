# A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks

This project is for the paper "[A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888)". Some codes are from [odin-pytorch](https://github.com/ShiyuLiang/odin-pytroch), [LID](https://github.com/xingjunm/lid_adversarial_subspace_detection), and [adversarial_image_defenses](https://github.com/facebookresearch/adversarial_image_defenses).

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requries Pytorch package to be installed:

* [Pytorch](http://pytorch.org/): Only GPU version is available.
* [scipy](https://github.com/scipy/scipy)
* [scikit-learn](http://scikit-learn.org/stable/)

## Downloading Out-of-Distribtion Datasets
We use download links of two out-of-distributin datasets from [odin-pytorch](https://github.com/ShiyuLiang/odin-pytorch):

* [Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* [LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

## Downloading Pre-trained Models
We provide six pre-trained neural networks (1) three DenseNets trained on [CIFAR-10](https://www.dropbox.com/s/pnbvr16gnpyr1zg/densenet_cifar10.pth?dl=0), [CIFAR-100](https://www.dropbox.com/s/7ur9qo81u30od36/densenet_cifar100.pth?dl=0) and [SVHN](https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth?dl=0), where models trained on CIFAR-10 and CIFAR-100 are from [odin-pytorch](https://github.com/ShiyuLiang/odin-pytroch), and (2) three ResNets trained on [CIFAR-10](https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth?dl=0), [CIFAR-100](https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth?dl=0) and [SVHN](https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth?dl=0). Please place them to `./pre_trained/`.
