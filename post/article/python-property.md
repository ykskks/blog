---
title: "PythonのPropertyの必要性を理解する"
date: 2020-08-19
tags: ["Python"]
draft: true
---
他人の書いたPythonコードを読んでいるとクラス内で`@property`などというものが使われているのをを見かけることがある。

以前にも何度か遭遇し、その度に調べてはなんとなく理解をしてやり過ごしていたが、結局本質的な理解ができてないということに気づいたので、今回ちゃんとまとめてみようと思った。

`@property`自体の使い方について説明した記事は多くあるが、そのモチベーションについて理解するための記事が少ない気がしたのでその部分に重点をおいて説明する。

まずこんな例を考えてみる。

あるサービスでは、ユーザが商品を購入すると、商品の値段に応じてポイントが貯まるようになっている。そのポイントの貯まり方は、ユーザの利用期間によって以下のように決まる。

- 3年以下の時は(商品値段)ポイント
- 5年以下の時は(1.2 * 商品値段)ポイント
- 6年以上は(1.5 * 商品値段)ポイント

これは例えば以下のように実装できる。

```
class NaiveUser:
    def __init__(self, name):
        self.name = name
        self.year = 0
        self.score = 0
        self.score_rate = 1

    def buy(self, price):
        print(f"Bought! Price={price}")
        self.score += int(self.score_rate * price)
```
score_rateとyearは別々に定義されていて、yearに変更が入るたびにscore_rateを更新する、という形式である。

しかし、本来score_rateはyearによってのみ決まるので、score_rateをyearにより決定されるpropertyとみなして

```
class BetterUser:
    def __init__(self, name):
        self.name = name
        self.year = 0
        self.score = 0

    @property
    def score_rate(self):
        if self.year <= 3:
            return 1
        elif self.year <=5:
            return 1.2
        else:
            return 1.5

    def buy(self, price):
        print(f"Bought! Price={price}")
        self.score += int(self.score_rate * price)
```
とした方がより安全である。

BetterUserでは、score_rateの書き換えができない。

```
u = User("Ben")
u.year = 9

print(u.score_rate)
>> 1.5

u.buy(300)
>> 450

u.score_rate = 1.2
>> AttributeError: can't set attribute
```

このように本来他の属性のみによって決定され、外側から変更されるべきでない属性を変更できないようにする、より一般的にはクラス属性へのアクセスを細かく制御するために`@property`は存在する。

## まとめ
`@property`を使えばクラス属性の細かな制御を実現できる。これにより、意図しない値の変更・消去などを防ぐことができる。





