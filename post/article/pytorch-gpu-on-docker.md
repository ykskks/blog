---
title: "DockerでPyTorch + CUDA の環境構築"
date: 2020-10-29
tags: ["環境構築"]

---
研究で必要になったので、Docker上でPyTorch + CUDAの環境を構築した時のメモ。

取りあえずこれで動くようになったが一部要検証の部分があるので、今後追記するかも。

## 環境
- Ubuntu18.04
- GeForce GTX 1080 Ti
- Docker 19.03

Dockerは既にインストール済みとします。

## NVIDIA driver
`ubuntu-drivers devices`で`recommended`を確認し`nvidia-450`と出た。従うなら

`sudo ubuntu-drivers autoinstall`
でOK

必要なCUDAのバージョンとの互換性を確認しておく

## CUDA toolkit
https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork

downloadしてinstructionに従う

```
sudo dpkg -i cuda-repo-ubuntu1804_10.1.105-1_amd64.deb`
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-10-1
```

```
nvidia-smi
>>> CUDA version 11.1

nvcc -V
>>> release 10.1
```
nvccとnvidia-smiのCUDAのバージョン表示にズレがあるが、どうやら問題ないらしい...

https://medium.com/@brianhourigan/if-different-cuda-versions-are-shown-by-nvcc-and-nvidia-smi-its-necessarily-not-a-problem-and-311eda26856c

```
sudo reboot

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
exit

sudo apt autoremove
sudo apt clean

```

conda使うならPyTorchのインストール時にcuda-toolkitも入れてくれるらしく、自分で入れなくてもよかったらしい（要検証）

- https://stackoverflow.com/questions/59529804/nvidia-cudatoolkit-vs-conda-cudatoolkit
- https://katsuwosashimi.com/archives/742/how-to-choose-cuda-version-pytorch/

## nvidia-container-toolkit

https://medium.com/nvidiajapan/nvidia-docker-%E3%81%A3%E3%81%A6%E4%BB%8A%E3%81%A9%E3%81%86%E3%81%AA%E3%81%A3%E3%81%A6%E3%82%8B%E3%81%AE-20-09-%E7%89%88-558fae883f44
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Dockerfileと環境の立ち上げ

https://github.com/ykskks/MF-SE/blob/master/README.md
