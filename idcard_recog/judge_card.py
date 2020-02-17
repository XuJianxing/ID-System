"""判断是不是身份证图片"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import tensorflow as tf
if __name__ == '__main__':
    import cnn_model
else:
    from idcard_recog import cnn_model


def find_face(image):
    '''返回是否人脸在图片右边
    输入：cv2彩图
    返回：是 否
    '''
    #face_cascade = cv2.CascadeClassifier('E:\\project\\verification\\datas\\haarcascades\\haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    if __name__ == "__main__":
        face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
    else:
        face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('C:\\Users\\Xu Jianxing\\Documents\\projects\\Card-Ocr-master\\datas\\haarcascades\\haarcascade_eye.xml')
    if len(image.shape) == 3 or len(image.shape) == 4:    # 判断是不是彩色图（通道数）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray = cv2.split(img)[0]    # 按B分灰度 (实验证明这样不好)
    else:
        gray = image
    #cv2.imshow("aa", gray)
    #cv2.waitKey()
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    #faces = face_cascade.detectMultiScale(gray)
    print("face count:", len(faces))
    if len(faces) > 0:    # 判断有没有脸，有的话再找。# 先找脸，再矫正；否则先矫正，再找脸（因为检测方法不好，容易漏检）
        (x,y,w,h) = faces[0]


