---
title: "DockerでDjango + Selenium の環境構築"
date: 2020-11-01
tags: ["環境構築"]

---
## 概要
cronを利用してSeleniumで定期的にスクレイピングをして情報を取得し、それらを表示するようなWebアプリケーションを作成した際に、Dockerで環境を構築したのでその時のメモ。

Docker初心者なので間違いはあるかもしれないが、とりあえずやりたいことは大体できました。

コードは一応ここにあります。

https://github.com/ykskks/ad-tracker

## 環境
macOS Catalina 10.15.7

Docker Desktop 2.1.0.1

## Dockerfile
```
FROM python:3

RUN apt-get update && apt-get install -y unzip

# google chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list' && \
    apt-get update && \
    apt-get install -y sudo && \
    apt-get install -y google-chrome-stable && \
    apt-get install -y python3-pip python3-dev libpq-dev postgresql postgresql-contrib

# chrome driver
RUN mkdir --mode=755 -p /opt/chrome
ADD https://chromedriver.storage.googleapis.com/83.0.4103.39/chromedriver_linux64.zip /opt/chrome
RUN cd /opt/chrome/ && \
    unzip chromedriver_linux64.zip

ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/chrome

COPY requirements.txt .

RUN pip install -U pip && \
    pip install -r requirements.txt
```

## docker-compose.yml
```
version: "3"
services:
  app:
    build: .
    volumes:
      - ${PWD}:/working
    working_dir: /working
    ports:
      - 8000:8000
    tty: true
```

## 環境立ち上げ
```
docker-compose up --build -d
docker-compose exec app bash
```

## 反省
環境を起動する度にまっさらの環境になるため、DBの設定やDBのデータ自体が吹っ飛んでしまう。

この辺りは「データの永続化」などのキーワードで調べればたくさん出てくるが今回はあくまでもデモだったので後回しにしてしまった。このあたり結構ややこしそうでなかなか大変そうだなと思った。