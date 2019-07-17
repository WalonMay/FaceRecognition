# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import detect_face#https://github.com/ShyBigBoy/face-detection-mtcnn
import numpy as np
import os

# Step 1:读取mtcnn网络
minsize = 40
threshold = [ 0.6, 0.7, 0.7 ]
factor = 0.709
gpu_memory_fraction=1.0
with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess1.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess1, None)
# Step 2:读取cnn网络
sess2 = tf.Session()
ckpt = tf.train.get_checkpoint_state('C:/Users/May-Walon/Desktop/model/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
saver.restore(sess2,ckpt.model_checkpoint_path)

graph = tf.get_default_graph()
x = graph.get_tensor_by_name('input_x_org:0')
SM = graph.get_tensor_by_name('SM:0')
# Step 3:读取对应列标签
target = list(os.walk('C:/Users/May-Walon/Desktop/faceImagesGray'))[0][1]
# Step 4:人脸识别
pic = cv2.VideoCapture(0)
while True:
    #读取照片，展示
    ret, frame = pic.read()
    frame = cv2.flip(frame, 1, dst=None)
    #cv2.imshow("Face Recognition", frame)
    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    #有时候在bounding_boxes里有多个对象，但第一个是人脸
    if bounding_boxes.shape[0] and bounding_boxes[0][0]>0 and bounding_boxes[0][1]>0:
        face_position = bounding_boxes[0]
        face_position=face_position.astype(int)
        cv2.rectangle(frame, (face_position[0], face_position[1]),
            (face_position[2], face_position[3]), (0, 255, 0), 2)
        #cv2.imshow("Face Recognition", frame)
        crop=frame[face_position[1]:face_position[3],
                 face_position[0]:face_position[2],]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        grayarray = np.array(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_CUBIC),dtype="float32").reshape(1,32,32)/255
        sm = sess2.run(SM,feed_dict={x:grayarray}).reshape(10)
        text = target[np.argmax(sm)] +": %.2f %%" % (np.max(sm)*100)
        cv2.putText(frame, text, (face_position[0]+10, face_position[1]+20), 
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 255, 0), 1, False)
        cv2.imshow("Face Recognition",frame)
    #退出
    input = cv2.waitKey(1) & 0xFF
    if input == ord('q'):#退出程序
        break
# Step 5:释放资源
pic.release()
cv2.destroyAllWindows()
sess1.close()
sess2.close()
#plt.imshow(grayarray.reshape(96,96), cmap=plt.cm.gray, interpolation='nearest')