---
title: "PyTorchのBatchSamplerでエポックごとにサンプリングをする"
date: 2020-11-04
tags: ["Python", "PyTorch"]

---
## 概要
不均衡データに対して学習させる際に、正例は固定して、負例をエポックごとにランダムサンプリングしたいということがあります。（ありました。）

より一般的には、あるデータセットに対して学習を行う際に、毎回同じデータを使うのではなく、エポックごとに何らかの基準によってサンプリングをしたい、という状況です。

他にも方法はあるかもしれませんが、とりあえず`BatchSampler`を使えば、PyTorchでこれを実現できました。参考になる資料が少なかったのでメモしておきます。

## 実装
ここでは、負例が多いようなデータから負例のみをサンプリングしたいという状況を想定します。

```
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler
```
簡単なDatasetの定義しておきます。
```
class MyDataset(Dataset):

    def __init__(self, X, y):
        self.out = []
        for X_i, y_i in zip(X, y):
            self.out.append((X_i, y_i))

    def __len__(self):
        return len(self.out)

    def __getitem__(self, idx):
        X_i, y_i = self.out[idx]
        return X_i, y_i
```
次にメインのBatchSamplerの実装です。
```
class MySampler(BatchSampler):
    """
    与えられたneg_sample_ratioに従って、負例をランダムサンプリングする
    """
    def __init__(self, dataset, neg_sample_ratio, batch_size):
        # 何度も必要になるデータセットの情報をここで計算しておく
        self.dataset_size = len(dataset)
        self.labels = torch.tensor(dataset.out)[:, 1]
        self.pos_idx = torch.arange(self.dataset_size)[self.labels == 1]
        self.neg_idx = torch.arange(self.dataset_size)[self.labels == 0]
        self.idx_selected = None
        self.neg_sample_ratio = neg_sample_ratio
        self.neg_sample_size = int(len(self.neg_idx) * self.neg_sample_ratio)
        self.batch_size = batch_size

    def __iter__(self):
        cnt = 0

        neg_idx_selected = self.neg_idx[torch.randperm(len(self.neg_idx))[:self.neg_sample_size ]]
        self.idx_selected = torch.cat([self.pos_idx, neg_idx_selected], axis=0)
        self.idx_selected = self.idx_selected[torch.randperm(len(self.idx_selected))]  # shuffle

        while (cnt+1) * self.batch_size < len(self.idx_selected):
            indices = self.idx_selected[cnt * self.batch_size: (cnt+1) * self.batch_size]
            yield indices
            cnt += 1

    def __len__(self):
        return len(self.idx_selected) // self.batch_size
```
`__iter__`が呼ばれる度に、予め定められた`neg_sample_ratio`に従って負例データをランダムにサンプリングし、正例データと合体してシャッフルしたのち、`batch_size`ごとに`indices`として返すような実装になっています。ここをやりたいことによってカスタマイズすれば良いです。

## 実例
簡単な例を載せておきます。

負例80%、正例20%の不均衡データで学習時に負例をサンプリングしたいような状況です。
```
X = torch.randn(100)
y = torch.tensor([0] * 80 + [1] * 20)
y = y[torch.randperm(len(y))]
```
以下のように、BatchSamplerを使わずにDataLoaderを定義すると、当然そのままの不均衡なデータが学習時に流れ込むことになります。
```
train_dataset = MyDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=10)

for X_batch, y_batch in train_loader:
    print(f"y is {y_batch}")
```
```
y is tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
y is tensor([0, 0, 0, 0, 0, 1, 0, 1, 0, 0])
y is tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y is tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
y is tensor([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
y is tensor([1, 1, 0, 0, 0, 1, 1, 1, 0, 0])
y is tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
y is tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
y is tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
y is tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
```
ここでBatchSamplerをDataLoaderに渡すと、
```
train_dataset = MyDataset(X, y)
train_sampler = MySampler(train_dataset, neg_sample_ratio=0.25, batch_size=10)
train_loader_sample = DataLoader(train_dataset, batch_sampler=train_sampler)

for X_batch, y_batch in train_loader_sample:
    print(f"Sampled y is {y_batch}")
```
```
Sampled y is tensor([1, 1, 0, 0, 1, 1, 1, 0, 0, 0])
Sampled y is tensor([1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
Sampled y is tensor([0, 0, 0, 0, 0, 1, 0, 1, 1, 1])
```
となり不均衡が解消されていることがわかります。