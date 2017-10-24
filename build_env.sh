#!/bin/bash

echo "Authorize creation"
chown ubuntu -R /mnt

echo "creating folder"
mkdir /mnt/tcc_levi

echo "Moving to tools"
cd /home/ubuntu/caffe/build/tools

echo "Generating training set"
GLOG_logtostderr=1 ./convert_imageset --resize_height=227 --resize_width=227  /home/ubuntu/tcc_levi/tcc_dataset/train/ /home/ubuntu/tcc_levi/tcc_dataset/train.txt /mnt/tcc_data/train_leveldb 1

echo "Generating validation db"
GLOG_logtostderr=1 ./convert_imageset --resize_height=227 --resize_width=227  /home/ubuntu/tcc_levi/tcc_dataset/val/ /home/ubuntu/tcc_levi/tcc_dataset/val.txt /mnt/tcc_data/val_leveldb 1

echo "Calculating mean"
./compute_image_mean /mnt/tcc_levi/train_lmdb /mnt/tcc_levi/train_lmdb/mean_image.binaryproto
