# -*- coding: utf-8 -*-
import cv2
import os

index = 1
pic = cv2.VideoCapture(0)

#目标目录（目录必须存在）
target_dir = 'C:/Users/May-Walon/Desktop/CCC/'

while True:
    #读取照片，展示
    ret, frame = pic.read()
    frame = cv2.flip(frame, 1, dst=None)
    #draw_1 = cv2.rectangle(frame, (100,100), (200,200), (0,255,0), 2)
    cv2.imshow("capture", frame)
    #根据键盘输入作对应动作
    input = cv2.waitKey(1) & 0xFF
    if input == ord('d') and index > 1:#删除上张图片
        index -= 1
        os.remove(target_dir + '/' + '%d.jpg' % index)
    elif input == ord('x'):#照相
        cv2.imwrite(target_dir + '/' + '%d.jpg' % index,
            cv2.resize(frame, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_AREA))
        print("%d 张图片" % index)
        index += 1
    if index == 601 or input == ord('q'):#退出程序
        break
pic.release()
cv2.destroyAllWindows()