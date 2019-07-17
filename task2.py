# -*- coding: utf-8 -*-
import imageio
import tensorflow as tf
import detect_face#https://github.com/ShyBigBoy/face-detection-mtcnn
import cv2
import os
#参数
minsize = 40 #脸矩阵最小值
threshold = [ 0.6, 0.7, 0.7 ]  #三步对应参数
factor = 0.709 #过滤因子
gpu_memory_fraction=1.0
#源目录、目标目录（目录必须存在）
image_dir = 'C:/Users/May-Walon/Desktop/faceImages'
target_dir = 'C:/Users/May-Walon/Desktop/faceImagesGray'
#创建网络+读取参数
with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
#对目录下图片进行人脸提取+灰度化
for subdir in list(os.walk(image_dir))[0][1]:
    #若目录不存在，创建目录
    if not os.path.exists(target_dir + "/" + subdir):
        os.makedirs(target_dir + "/" + subdir)
    #提取+灰度化流程
    for i in range(1,601):
        img = imageio.imread(image_dir + "/" + subdir + '/%d.jpg' % i)            
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        
        #有时候在bounding_boxes里有多个对象，但第一个是人脸
        if bounding_boxes.shape[0]:
            face_position = bounding_boxes[0]
            face_position=face_position.astype(int)
            cv2.rectangle(img, (face_position[0], face_position[1]),
                (face_position[2], face_position[3]), (0, 255, 0), 2)
            crop=img[face_position[1]:face_position[3],
                     face_position[0]:face_position[2],]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(target_dir + "/" + subdir + '/%d.jpg' % i,
                cv2.resize(gray,gray.shape,interpolation = cv2.INTER_AREA))
        else:
            print(image_dir + "/" + subdir + '/%d.jpg' % i + " can't find faces!")#输出非人脸图片路径
