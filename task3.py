# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.image as mpimg
#源目录、目标目录（目录必须存在）
image_dir = 'C:/Users/May-Walon/Desktop/faceImagesGray/'
target_dir = 'C:/Users/May-Walon/Desktop/'

dataset = np.zeros((6000,32,32))
train_data = np.zeros((4800,32,32))
test_data = np.zeros((1200,32,32))
label = ['' for i in range(6000)]
#对目录下图片进行人脸提取+灰度化
index = 0
for subdir in list(os.walk(image_dir))[0][1]:
    #提取+灰度化流程
    for i in range(1,601):
        img = mpimg.imread(image_dir + subdir + '/%d.jpg' % i)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        dataset[index,:,:] = np.asarray(img, dtype='float64')#储存矩阵
        label[index] = subdir
        index += 1

#8:2划分训练集和测试集
sort_index = np.array([i for i in range(600)])
np.random.shuffle(sort_index)
for i in range(10):
    train_data[480*i:480*(i+1),:,:] = dataset[ sort_index[:480]+600*i ,:,:]/255
    test_data[120*i:120*(i+1),:,:] = dataset[ sort_index[480:]+600*i ,:,:]/255
train_label = np.array(list(os.walk(image_dir))[0][1]).repeat(480,axis=0).reshape(4800,1)
test_label = np.array(list(os.walk(image_dir))[0][1]).repeat(120,axis=0).reshape(1200,1)
#保存数据到本地文件
np.save(target_dir + 'train_data',train_data)
np.save(target_dir + 'test_data',test_data)
np.save(target_dir + 'train_label',train_label)
np.save(target_dir + 'test_label',test_label)