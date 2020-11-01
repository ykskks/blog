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

## Dockerfile
rdkitは化合物をpythonで扱うライブラリですが、こちらのインストールがcondaが一番楽なので今回はcondaのベースイメージを使っています。
```
FROM continuumio/anaconda3:2020.02

# ライブラリの追加インストール
RUN pip install -U pip && \
    pip install fastprogress japanize-matplotlib && \
    conda install -c conda-forge rdkit && \
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch && \
    pip install jupyter_contrib_nbextensions && \
    pip install streamlit

ARG UID
ARG GID
ARG UNAME

ENV UID ${UID}
ENV GID ${GID}
ENV UNAME ${UNAME}

RUN groupadd -g ${GID} ${UNAME}
RUN useradd -u ${UID} -g ${UNAME} -m ${UNAME}
```
## docker-compose.yml
runtimeをnvidiaに設定する
```
version: "2.3"
services:
  dev:
    user: $UID:$GID
    build:
      context: .
      args:
        UID: $UID
        GID: $GID
        UNAME: $UNAME
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    volumes:
      - ${PWD}:/working
    working_dir: /working
    ports:
      - 8888:8888
    command: jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
    tty: true
```

## 環境の立ち上げ
コンテナ内ではデフォルトでrootユーザとなり、コンテナ内でファイル作成した際などにホストからアクセスできなくなるなどの権限周りのトラブルが発生するのでユーザをホストに合わせる。

```
#!/bin/sh

touch .env
echo "UID=$(id -u $USER)" >> .env
echo "GID=$(id -g $USER)" >> .env
echo "UNAME=$USER" >> .env
```

これを実行してから環境を立ち上げてコンテナ内に入る

```
chmod 755 ./setDotEnv.sh
./setDotEnv.sh

docker-compose up --build -d
docker-compose exec dev bash
```

## 反省
Docker使ってるのに結局環境を一元管理できておらず、かなりホスト側の環境に依存してるので、この辺もう少しうまくできる方法ないかなーという感じ...