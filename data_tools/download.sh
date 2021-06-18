#!/bin/sh
wget -P /opt/ml/input/data/ https://prod-aistages-public.s3.ap-northeast-2.amazonaws.com/app/Competitions/000043/data/train_dataset.zip
cd /opt/ml/input/data
unzip train_dataset.zip
cd ~
