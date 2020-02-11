FROM nvidia/cuda:10.0-cudnn7-devel
MAINTAINER Sungdae <scar9cube@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive
RUN cd /etc/apt && \
       sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' sources.list
RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y build-essential \
    openjdk-8-jdk \
    language-pack-ko \
    fonts-nanum \
    fonts-nanum-coding \
	python3-dev \
	python3-pip \
    zip \
	&& apt-get clean

RUN locale-gen ko_KR.UTF-8
ENV LANG ko_KR.UTF-8
ENV LANGUAGE ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

COPY ceslea.zip /home/
WORKDIR /home
RUN unzip ./ceslea.zip && rm ./ceslea.zip
RUN mkdir /root/.pip/ && mv ./pip.conf /root/.pip/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
#RUN pip3 uninstall sanic -y
#RUN pip3 install sanic==18.12.0
#RUN rasa train --data /home/src/rasachat_src/data --config /home/src/rasachat_src/config.yml --domain /home/src/rasachat_src/domain.yml
WORKDIR /home/src/BERT_Compressed
#RUN python3 main.py
#RUN rasa train
CMD python3 main.py

EXPOSE 50051
