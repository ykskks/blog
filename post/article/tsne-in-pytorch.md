---
title: "PyTorchでt-SNEを実装"
date: 2020-09-17
tags: ["機械学習", "Python", "PyTorch"]

---

## 概要
高次元データの可視化によく用いられるt-SNEの動作を理解するために、論文を読んで実装してみることにしました。パラメータ更新時の勾配計算で楽をしたいのでPyTorchで実装します。

基本的には[元論文](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)を参考に実装を行いました。

今回使ったコードは[Github]()にあげています。

**注意!!**: 自分で実装して理解することが目的であったため動作は**劇遅**で実用的ではありません。何かお気づきの方はぜひ教えてください。

## SNE
t-SNEには前身であるSNEなる手法が存在し、t-SNEはSNEの弱点を補った手法です。そこでまずはSNEを実装します。

詳細は論文を参考して頂きたいのですが、大まかな流れは以下の通り。

- 入力データ$$X$$(N x d行列)
- 指定するパラメータは主にn_componentsとperplexity。前者は圧縮したい次元数で後者は後ほど説明。
- 出力は低次元に圧縮されたデータ$$y$$(N x n_components行列)


1. $$y$$をランダムに初期化
2. 高次元空間の各データポイントに対応する正規分布の分散を指定されたperplexityから求める。
3. 高次元空間における各データポイント間の類似性を求める。
4. 収束するまで以下を繰り返し

    - 低次元空間における各データポイント間の類似性を求める。
    - 高次元空間と低次元空間における類似性が近づく方向へ$$y$$を更新

***

`perplexity`ですが、高次元における各データポイントの類似性($$p_{j|i}$$を自分以外の全ての$$j$$について求めたもの)のシャノンエントロピーとして定義されています。

つまりは、SNEは高次元の類似性を低次元でも保つように低次元表現を学習しますが、その高次元の類似性を算出する際に各データポイントの近傍をどれくらいまで考慮するのか、ということを調節していると考えられます。

極端に考えれば、`perplexity`をめちゃくちゃ小さくすると各データポイントの類似性のエントロピーが小さいことを意味するので、対応する正規分布の分散は小さいものに設定している、つまり本当に近傍にあるデータポイントのみを考慮して類似性を算出していると考えられます。

詳しくは論文を参照ください。

***

```
class SNE:
    def __init__(self, n_components, perplexity, lr=1.0, n_epochs=100):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_epochs = n_epochs

    def _compute_perplexity_from_sigma(self, data_matrix, center_idx, sigma):
        similarities = torch.zeros(self.N)
        for i in range(self.N):
            similarities[i] = self._similarity(data_matrix[center_idx, :], data_matrix[i, :], sigma)
        p = torch.zeros(self.N)
        for i in range(self.N):
            p[i] = similarities[i] / similarities.sum()
        shannon = - (p[p != 0] * torch.log2(p[p != 0])).sum()  # log0=nan回避
        perp = 2 ** shannon.item()
        return perp

    def _search_sigmas(self, data_matrix):
        sigmas = torch.zeros(self.N)
        sigma_range = np.arange(0.1, 0.6, 0.1)
        for i in tqdm(range(self.N), desc="search sigma"):
            perps = np.zeros(len(sigma_range))
            for j, sigma in enumerate(sigma_range):
                perp = self._compute_perplexity_from_sigma(data_matrix, i, sigma)
                perps[j] = perp
            best_idx = (np.abs(perps - self.perplexity)).argmin()
            best_sigma = sigma_range[best_idx]
            sigmas[i] = best_sigma
        # print(f"Selected sigmas are {sigmas}")
        return sigmas

    def _similarity(self, x1, x2, sigma):
        return torch.exp(- ((x1 - x2) ** 2).sum() / 2 * (sigma ** 2))

    def _compute_similarity(self, data_matrix, sigmas):
        similarities = torch.zeros((self.N, self.N))
        for i, j in product(range(self.N), range(self.N)):
            g_ji = self._similarity(data_matrix[i, :], data_matrix[j, :], sigmas[i])
            similarities[i][j] = g_ji
        return similarities

    def _compute_cond_prob(self, similarities):
        cond_prob_matrix = torch.zeros((self.N, self.N))
        for i, j in product(range(self.N), range(self.N)):
            p_ji = similarities[i][j] / similarities[i].sum()
            cond_prob_matrix[i][j] = p_ji
        return cond_prob_matrix

    def fit_transform(self, X):
        self.N = X.shape[0]
        X = torch.tensor(X)

        # 1. yをランダムに初期化
        y = torch.randn(size=(self.N, self.n_components), requires_grad=True)
        optimizer = optim.Adam([y], lr=self.lr)

        # 2. 高次元空間の各データポイントに対応する正規分布の分散を指定されたperplexityから求める。
        sigmas = self._search_sigmas(X)

        # 3. 高次元空間における各データポイント間の類似性を求める。
        X_similarities = self._compute_similarity(X, sigmas)
        p = self._compute_cond_prob(X_similarities)

        # 4. 収束するまで以下を繰り返し
        loss_history = []
        for i in tqdm(range(self.n_epochs), desc="fitting"):
            optimizer.zero_grad()
            y_similarities = self._compute_similarity(y, torch.ones(self.N) / (2 ** (1/2)))
            q = self._compute_cond_prob(y_similarities)

            kl_loss = (p[p != 0] * (p[p != 0] / q[p != 0]).log()).sum()  # log0=nan回避
            kl_loss.backward()
            loss_history.append(kl_loss.item())
            optimizer.step()

        plt.plot(loss_history)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        return y.detach().numpy()
```

### 注意点
- 論文では、正規分布の分散を二部探索で求めているとの記述がありましたが、上の実装では単に範囲を決め打ちして最も指定のperplexityに近いものを選んでいます。

### 結果
digitsデータを使ってPCAとSNEによる二次元への次元圧縮の様子を見てみます。

```
from itertools import product

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
```
digits = load_digits()
X, y = digits.data[:100, :], digits.target[:100]
print(X.shape, y.shape)
>> (100, 64) (1797,)
```
計算が重いのでサンプリングしてます。

まずはPCA
```
pca = PCA(n_components=2, random_state=42)
sc = StandardScaler()

X_sc = sc.fit_transform(X)
X_pca = pca.fit_transform(X_sc)

fig, ax = plt.subplots(figsize=(6, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple']
for c, label in zip(colors, digits.target_names):
    ax.scatter(X_pca[y == int(label), 0], X_pca[y == int(label), 1], color=c, label=label)
ax.legend()
ax.set_title("PCA", fontsize=16)
```

![](/article/img/digits_pca.png)

3と5が大部分被ってしまっていたり、まとまりをうまく捉えられていないところが見受けられます。

次にSNEですが

```
sne = SNE(n_components=2, perplexity=50, n_epochs=100, lr=1)
X_sne = sne.fit_transform(X_sc)

fig, ax = plt.subplots(figsize=(6, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple']
for c, label in zip(colors, digits.target_names):
    ax.scatter(X_sne[y == int(label), 0], X_sne[y == int(label), 1], color=c, label=label)
ax.legend()
ax.set_title("SNE", fontsize=16)
```

![](/article/img/sne_loss.png)

50イテレーションくらいで大体収束してるのがわかります。

![](/article/img/digits_sne.png)

2が若干広く伸びてしまっていますが、全体としてはPCAよりも各数字がギュッとまとまっており、データの構造をよく捉えているように見えます。

## t-SNE
いよいよt-SNEですが、SNEまで実装してしまえば、大したことはありません。

主な変更点は以下の二つです。（詳細は論文を参照願います）

- symmetricな類似性の導入
- 低次元空間の類似性を正規分布ではなく、裾の重い自由度1のt分布にする

***

コードです。メソッド名に`_hd`がついているものは高次元空間での計算に用いるもの、`_ld`は低次元空間での計算に用いるものです。

若干重複するコードが多く、あまり美しくないですが...

```
class TSNE:
    def __init__(self, n_components, perplexity, lr=0.01, n_epochs=100):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_epochs = n_epochs

    def _compute_perplexity_from_sigma(self, data_matrix, center_idx, sigma):
        similarities = torch.zeros(self.N)
        for i in range(self.N):
            similarities[i] = self._similarity_hd(data_matrix[center_idx, :], data_matrix[i, :], sigma)
        p = torch.zeros(self.N)
        for i in range(self.N):
            p[i] = similarities[i] / similarities.sum()
        shannon = - (p[p != 0] * torch.log2(p[p != 0])).sum()  #ゼロがlogとるとnanになるので省く
        perp = 2 ** shannon.item()
        return perp

    def _search_sigmas(self, data_matrix):
        sigmas = torch.zeros(self.N)
        sigma_range = np.arange(0.1, 0.6, 0.1)
        for i in tqdm(range(self.N), desc="search sigma"):
            perps = np.zeros(len(sigma_range))
            for j, sigma in enumerate(sigma_range):
                perp = self._compute_perplexity_from_sigma(data_matrix, i, sigma)
                perps[j] = perp
            best_idx = (np.abs(perps - self.perplexity)).argmin()
            best_sigma = sigma_range[best_idx]
            sigmas[i] = best_sigma
        # print(f"Selected sigmas are {sigmas}")
        return sigmas

    def _similarity_hd(self, x1, x2, sigma):
        return torch.exp(- ((x1 - x2) ** 2).sum() / 2 * (sigma ** 2))

    def _similarity_ld(self, x1, x2):
        return (1 + ((x1 - x2) ** 2).sum()) ** (-1)

    def _compute_similarity_hd(self, data_matrix, sigmas):
        similarities = torch.zeros((self.N, self.N))
        for i, j in product(range(self.N), range(self.N)):
            g_ji = self._similarity_hd(data_matrix[i, :], data_matrix[j, :], sigmas[i])
            similarities[i][j] = g_ji
        return similarities

    def _compute_similarity_ld(self, data_matrix):
        similarities = torch.zeros((self.N, self.N))
        for i, j in product(range(self.N), range(self.N)):
            g_ji = self._similarity_ld(data_matrix[i, :], data_matrix[j, :])
            similarities[i][j] = g_ji
        return similarities

    def _compute_cond_prob_hd(self, similarities):
        cond_prob_matrix = torch.zeros((self.N, self.N))
        for i, j in product(range(self.N), range(self.N)):
            p_ji = similarities[i][j] / similarities[i].sum()
            cond_prob_matrix[i][j] = p_ji

        cond_prob_matrix = (cond_prob_matrix + torch.t(cond_prob_matrix)) / 2
        return cond_prob_matrix

    def _compute_cond_prob_ld(self, similarities):
        cond_prob_matrix = torch.zeros((self.N, self.N))
        for i, j in product(range(self.N), range(self.N)):
            p_ji = similarities[i][j] / similarities[i].sum()
            cond_prob_matrix[i][j] = p_ji
        return cond_prob_matrix

    def fit_transform(self, X):
        self.N = X.shape[0]
        X = torch.tensor(X)

        # init low dimentional y
        y = torch.randn(size=(self.N, self.n_components), requires_grad=True)
        optimizer = optim.Adam([y], lr=self.lr)

        sigmas = self._search_sigmas(X)
        X_similarities = self._compute_similarity_hd(X, sigmas)
        p = self._compute_cond_prob_hd(X_similarities)
        loss_history = []
        for i in tqdm(range(self.n_epochs), desc="fitting"):
            optimizer.zero_grad()
            y_similarities = self._compute_similarity_ld(y)
            q = self._compute_cond_prob_ld(y_similarities)

            kl_loss = (p[p != 0] * (p[p != 0] / q[p != 0]).log()).sum()  # 対角成分のゼロがlogとるとnanになる
            kl_loss.backward()
            loss_history.append(kl_loss.item())
            optimizer.step()

        plt.plot(loss_history)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        return y.detach().numpy()
```

### 結果

```
tsne = TSNE(n_components=2, perplexity=50, n_epochs=200, lr=1)
X_tsne = tsne.fit_transform(X_sc)

fig, ax = plt.subplots(figsize=(6, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple']
for c, label in zip(colors, digits.target_names):
    ax.scatter(X_tsne[y == int(label), 0], X_tsne[y == int(label), 1], color=c, label=label)
ax.legend()
ax.set_title("TSNE", fontsize=16)
```

![](/article/img/tsne_loss.png)

![](/article/img/digits_tsne.png)

どうでしょうか？

正直各グループのまとまり具合はそんなに変わらないかなーという感じがしますが、
グループ間の距離が少し空いていて、より見た目がすっきりした感があります。（SNEはごちゃごちゃ全体が固まっている感じ、論文でcrowding problemと呼ばれているもの？）

perplexityの値は割と敏感らしいのでもう少し調節しても良いと思いますが、とりあえずそれっぽい結果が出ているのでとりあえずここまでにしようと思います。

***

（ちなみに）

2の中に他の2たちからかなり離れている奴らがいますが、見てみるとこんな感じでした。上の段の左から2番目と3番目です。確かに他に比べて癖があるようです。9に近いのも納得できそうです。

![](/article/img/2.png)

![](/article/img/9.png)


## TODO
- 高速化(高次元空間で遠いデータポイント間の類似性をゼロにしてしまうなど近似手法があるようです...)

多分また書きます。

何か間違い等ありましたら遠慮なくご指摘ください。


