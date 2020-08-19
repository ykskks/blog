---
title: "PythonのPropertyの必要性を理解する"
date: 2020-08-19
tags: ["Python"]

---
他人の書いたPythonコードを読んでいるとクラス内で`@property`などというものが使われているのをを見かけることがある。

以前にも何度か遭遇し、その度に調べてはなんとなく理解をしてやり過ごしていたが、結局本質的な理解ができてないということに気づいたので、今回ちゃんとまとめてみようと思った。

（なお、筆者はPythonでプログラミングを始めたので、他の記事にあるような他のプログラミング言語との比較はあまり理解に役立たなかった。そのような背景の人もいるのではないかと思い、ここでまとめてみる。）

`@property`自体の使い方について説明した記事は多くあるが、そのモチベーションについて理解するための記事が少ない気がしたのでその部分に重点をおいて説明する。

まずこんな例を考えてみる。

```
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

実際には人の名前は変わることもあると思うが、ここでは名前は本来変化しないものであると仮定する。年齢は当然変化する。

一方で現在の実装では以下のようなことができてしまう。

```
person = Person("Mark", 34)
print(person.name)
>> Mark

person.name = "Bob"
print(person.name)
>> Bob
```
このように本来外側から変更されるべきでない属性が変更されてしまうといったことを防ぐため、より一般的にはクラス属性へのアクセスを細かく制御するために`@property`は存在する。

この例の場合以下のような制御を行いたい。

- `name`は変更できない
- `age`は現在の値より大きい値にのみ変更できる

これは以下のように実現できる。

```
class Person:
    def __init__(self, name, age):
        self._name = name
        self._age = age

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        raise ValueError("Name cannot be changed.")

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, age):
        if age < self._age:
            raise ValueError("Age cannnot be set as smaller value than current value.")
        self._age = age
```

このような実装をすれば、想定外のことが起こるのを防ぐことができる。

```
person = Person("Mark", 34)
print(person.name)
>> Mark
print(person.age)
>> 34

# 名前の変更はできない
person.name = "Bob"
>> ValueError: Name cannot be changed.

person.age = 35
print(person.age)
>> 35

# 年齢は大きい値にしか変更できない。
person.age = 33
>> ValueError: Age cannnot be set as smaller value than current value.
```

なお、Pythonにはプライベート変数が存在しないので実際には

```
person._name = "Bob"
print(person.name)
>> Bob
```
のように変更できてしまう。

しかし、アンダースコアで始まる変数はプライベートということにしよう、というのが`お約束`なので、それが守られることを仮定すれば、少なくとも意図しないような属性の書き換えが起こってしまうようなことは避けられるんじゃないかなー、という感じ。

## まとめ
`@property`を使えばクラス属性の細かな制御を実現できる。これにより、意図しない値の変更・消去などを防ぐことができる。





