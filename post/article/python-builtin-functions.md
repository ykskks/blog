---
title: "Pythonの地味な組み込み関数たち"
date: 2020-08-18
tags: ["Python"]

---

Pythonの組み込み関数のうち、頻繁には使わない・ぱっと見では使い方がわからない「地味な」組み込み関数たちを勝手に選んでまとめていく記事です。（随時追加）

## dir()
- 引数がない時は、現在のローカルスコープの名前のリストを返す
- 引数がある時は、そのオブジェクトの属性のリストを返そうとする

## type()
- 引数が一つの時は、そのオブジェクトの型を返す

## max()
- key引数で「何を基準にmaxとするのか」を定義できる

例えば
```
nums = [-2, 0, 1]
def my_func(x): return x ** 2 - 1

print(max(nums))
>> 1

print(max(nums, key=my_func))
>> -2
```

辞書の最大のvalueに対応するkeyを取得したい際にもmax関数のkeyを使えば簡単に記述できる。
[参考のstackoverflow](https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary)


