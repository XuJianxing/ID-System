# -*- encoding:utf-8 -*-
'''
主程序 读图-旋转-分割-识别 于一体
传入图片地址，输出解析后字符串
'''
"""
version 1:
# blackhat -> close -> threshold -> close -> erode
检测定位与hough旋转；形态学定位；投影分割；识别
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import re
import sys
import cv2
import json
import random
import numpy as np
import tensorflow as tf
if __name__ == '__main__':
    import cnn_model
else:
    from idcard_recog import cnn_model

print(os.getcwd())

def by_first_down(t):
    '''sorted里按元祖第一个数值降序排列'''
    return -t[0]
def by_second_down(t):
    '''sorted里按元祖第二个数值降序排列'''
    return -t[1]


def find_face_2_rotatedcard(image):
    '''返回按脸找到的旋转矫正后的卡片位置图片：
    输入：cv2彩图
    返回：灰度IDcard区域图
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
        (x1,y1,w1,h1) = (int(x-0.1*w), int(y+1.2*h), int(1.2*w), int(1.5*h))
        word = gray[y1:y1+h1, x1:x1+w1].copy()
        #'''------rotate the pic------'''
        ''' # 旧方法
        ro_gray = cv2.bitwise_not(word)    # 像素值取反，不找最小矩形框就不用了
        thresh = cv2.threshold(ro_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]    # 算角度
        '''
        edges = cv2.Canny(word, 50, 150, apertureSize = 3)    # canny边缘检测
        lines = cv2.HoughLines(edges, 1, np.pi/180, 118)    #这里对最后一个参数使用了经验型的值118; 霍夫找直线
        #print("lines shape:", lines.shape)
        try:    # 如果lines是none的话，从原图里找线
            for line in lines:
                theta = line[0][1]    #第0条线;第0个不知道什么;第下标1个参数 (rho, theta)
                theta = theta * 180 / np.pi - 90
                if theta >= -45 and theta <= 45:
                    break
        except:    # 从原图里找线
            edges = cv2.Canny(word, 50, 150, apertureSize = 3)    # canny边缘检测
            lines = cv2.HoughLines(edges, 1, np.pi/180, 118)    #这里对最后一个参数使用了经验型的值118; 霍夫找直线
            try:
                for line in lines:
                    theta = line[0][1]    #第0条线;第0个不知道什么;第下标1个参数 (rho, theta)
                    theta = theta * 180 / np.pi - 90
                    if theta >= -45 and theta <= 45:
                        break
            except:
                theta = 0
        (h, w) = gray.shape[:3]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, theta, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        #------rotate ended, find card area------
        faces = face_cascade.detectMultiScale(rotated, 1.3, 5)
        (x,y,w,h) = faces[0]
        (x,y,w,h) = (int(x-3.5*w), int(y-0.8*h), 5*w, int(3.1*h))    # card area
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if w > rotated.shape[1]:
            w = rotated.shape[1]
        if h > rotated.shape[0]:
            h = rotated.shape[0]
        card = rotated[y:y+h, x:x+w].copy()
        return card
    else:    # 否则先矫正，再找脸
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)    # canny边缘检测
        lines = cv2.HoughLines(edges, 1, np.pi/180, 118)    #这里对最后一个参数使用了经验型的值118; 霍夫找直线
        for line in lines:
            theta = line[0][1]    #第0条线;第0个不知道什么;第下标1个参数 (rho, theta)
            theta = theta * 180 / np.pi - 90
            if theta >= -45 and theta <= 45:
                break
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, theta, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        #------rotate ended, find card area------
        faces = face_cascade.detectMultiScale(rotated, 1.3, 5)
        if len(faces) > 0:    # 如果能找到脸就找，还找不到就没办法了，只能返回原图
            (x,y,w,h) = faces[0]
            (x,y,w,h) = (int(x-3.5*w), int(y-0.8*h), 5*w, int(3.1*h))    # card area
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if w > rotated.shape[1]:
                w = rotated.shape[1]
            if h > rotated.shape[0]:
                h = rotated.shape[0]
            card = rotated[y:y+h, x:x+w].copy()
            return card
        else:
            print("no face")
            return gray

def find_emblem_2_rotatedcard(image):
    '''返回按国徽找到的旋转矫正后的卡片图片
    返回彩色图像，为了增加识别正确率'''
    emblem_cascade = cv2.CascadeClassifier("./haarcascades/cascade.xml")
    if len(image.shape) == 3 or len(image.shape) == 4:    # 判断是不是彩色图（通道数）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray = cv2.split(img)[0]    # 按B分灰度 (实验证明这样不好)
    else:
        gray = image
    
    emblems = emblem_cascade.detectMultiScale(gray, 1.3, 5)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    print("emblem count:", len(emblems))

    if len(emblems) > 0:    # 判断有没有图标，有的话再找。# 先找图标再矫正；否则先矫正，再找图标
        (x,y,w,h) = emblems[0]
        #------------找国徽里的直线（找的不准，先从全图里找 暂定）---------------------------
        gray = cv2.Canny(gray, 50, 150, apertureSize = 3)
        lines = cv2.HoughLines(gray, 1, np.pi/180, 118)    #这里对最后一个参数使用了经验型的值118; 霍夫找直线
        try:    # 如果lines是none的话，就不旋转了
            i = 0
            while i < len(lines) and (lines[i][0][1] < np.pi/4 or lines[i][0][1] > 3*np.pi/4):
                i += 1
                pass
            #第0条线；第0个不知道什么；第1个参数(rho or theta)
            theta = lines[i][0][1]
            theta = theta * 180 / np.pi
        except:    # 从原图里找线
            theta = 90
        print("theta", theta)
        theta = theta - 90
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, theta, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        emblems = emblem_cascade.detectMultiScale(rotated, 1.3, 5)  # 正斜找的框不一样大，所以要再找一次
        (x,y,w,h) = emblems[0]
        (x,y,w,h) = (int(x+1.2*w), int(y+1.4*h), int(3*w), int(1.4*h))    # card area
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if w > rotated.shape[1]:
            w = rotated.shape[1]
        if h > rotated.shape[0]:
            h = rotated.shape[0]
        card = rotated[y:y+h, x:x+w].copy()
        return card
    else:    # 没有就先先矫正，再找图标
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)    # canny边缘检测
        lines = cv2.HoughLines(edges, 1, np.pi/180, 118)    #这里对最后一个参数使用了经验型的值118; 霍夫找直线
        #for line in lines[0]: # 先只看lines里第一个直线
        #rho = lines[0][0] #第一个元素是距离rho
        i = 0
        while i < len(lines) and (lines[i][0][1] < np.pi/4 or lines[i][0][1] > 3*np.pi/4):
            i += 1
            pass
        theta = lines[i][0][1]    #第二个元素是角度theta：第0条线，第0个不知道什么，第1个参数(rho, theta)
        theta = theta * 180 / np.pi
        theta = theta - 90
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, theta, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        #------rotate ended, find card area------
        emblems = emblem_cascade.detectMultiScale(rotated, 1.3, 5)
        if len(emblems) > 0:    # 如果能找到脸就找，还找不到就没办法了，只能返回原图
            (x,y,w,h) = emblems[0]
            (x,y,w,h) = (int(x+1.2*w), int(y+1.4*h), int(3*w), int(1.4*h))    # card area
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if w > rotated.shape[1]:
                w = rotated.shape[1]
            if h > rotated.shape[0]:
                h = rotated.shape[0]
            card = rotated[y:y+h, x:x+w].copy()
            return card
        else:
            print("no emblem")
            return gray

def find_text(image, cut_type):
    '''
    输入传进的原图片和切割类型
    返回图片列表:
    1：地址, 号码, 姓名, 性别, 民族, (年,月,日), 头像图片
    2：签发机关，有效日期
    '''
    if cut_type == 1:
        image = find_face_2_rotatedcard(image)
        if image.shape[0] > 500:    # shape里是高，宽（行数，列数）
            image = cv2.resize(image, (int(500/image.shape[0]*image.shape[1]), 500))   # opencv里的顺序是宽、高
    else:
        image = find_emblem_2_rotatedcard(image)
        print("image.shape:", image.shape)
        if image.shape[0] > 200:
            #image = imutils.resize(image, height=200)
            image = cv2.resize(image, (image.shape[1],200))
        image = cv2.resize(image, (2*image.shape[0],image.shape[0]))

    print("image.shape:", image.shape)
    # 初始化一个矩形和正方形核
    if cut_type == 1:
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))    #矩形框(13, 5)
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))    # (21, 21)
    else:
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))    #矩形框(18, 8)（越大腐蚀越厉害，字范围越小）
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))    # (21, 21)

    # 用3*3高斯模糊来平滑图像，然后用黑帽形态学算子?来找到浅色背景下的深色区域
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# 因为image已经是灰度图了,用gray找区域Roi,然后分割image;灰度化以后图片变成单通道
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)    # 高斯模糊 滤波（平滑）
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, rectKernel)    # 消除缝隙
    
    # 计算blackhat图像的Scharr梯度然后规模到[0, 255]
    # 构造灰度图在水平和竖直方向上的梯度幅值表示。
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)    # 计算绝对值
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # 用矩形核应用闭合操作来关闭字母间的间隙  --  最大类间方差阈值方法
    # 这里进行形态学操作，将上一步得到的内核应用到我们的二值图中，以此来消除竖杠间的缝隙。？
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    # threshold二值化，需要输入单通道图片才能做阈值化
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 执行另一个闭合操作，这回用正方形核来消除机读区间的间隙，然后执行一系列腐蚀分开连接元素
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)    # 仅腐蚀？

    # 在阈值过程中有可能边界像素被包含在阈值中，
    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0
    # find contours in the thresholded image and sort them by their
    # size
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,   #找轮廓
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  #按大小排序
    # loop over the contours
    ro = []
    if cut_type == 1:
        ww = []
        hh = []
        xx = []
        yy = []
        # 加一个按高度h来判断，高的是地址。因为直接按大小顺序排可能会反，不按x排可能图片是倒立的
        (x, y, w, h) = cv2.boundingRect(cnts[0])
        (x1, y1, w1, h1) = cv2.boundingRect(cnts[1])
        if h1 > h:    # 第二个大于第一个才交换，否则不用改
            (x_, y_, w_, h_) = (x, y, w, h)
            (x, y, w, h) = (x1, y1, w1, h1)
            (x1, y1, w1, h1) = (x_, y_, w_, h_)
        # 计算等高线的轮廓，利用等高线计算轮廓宽度与图像宽度的长宽比和覆盖率。
        pX = int((x + w) * 0.03)
        pY = int((y + h) * 0.03)
        (x, y) = (x - pX, y - pY)
        (w, h) = (w + (pX * 2), h + (pY * 2))
        #print(x, y, w, h)
        ww.append(w)
        hh.append(h)
        xx.append(x)
        yy.append(y)
        roi = image[y:y + h, x:x + w].copy()
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ro.append(roi)    # 地址
        pX = int((x1 + w1) * 0.03)
        pY = int((y1 + h1) * 0.03)
        (x, y) = (x1 - pX, y1 - pY)
        (w, h) = (w1 + (pX * 2), h1 + (pY * 2))
        #print(x, y, w, h)
        ww.append(w)
        hh.append(h)
        xx.append(x)
        yy.append(y)
        roi = image[y:y + h, x:x + w].copy()
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ro.append(roi)    # 号码
    else:
        r = []    # 存用来排序的(x, y, w, h)
        for c in cnts:
            # 计算等高线的轮廓，利用等高线计算轮廓宽度与图像宽度的长宽比和覆盖率。
            (x, y, w, h) = cv2.boundingRect(c)

            ar = w / float(h)    # 宽高比
            #crWidth = w / float(gray.shape[1])     #and crWidth < 0.4
            '''# 待定，可以用xy坐标来规定区域'''
            if ar > 5:
                pX = int((x + w) * 0.04)
                pY = int((y + h) * 0.03)
                (x, y) = (x - pX, y - pY)
                (w, h) = (w + (pX * 2), h + (pY * 2))
                r.append((x, y, w, h))
                #roi = image[y:y + h, x:x + w].copy()
                #cv2.imshow("roi", roi)
                #cv2.waitKey(0)
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #ro.append(roi)
        # 先按y再按x排，原因暂不明
        r = sorted(r, key=by_second_down)
        r = sorted(r, key=by_first_down)
        
        roi = image[r[0][1]:r[0][1] + r[0][3], r[0][0]:r[0][0] + r[0][2]].copy()
        ro.append(roi)
        roi = image[r[1][1]:r[1][1] + r[1][3], r[1][0]:r[1][0] + r[1][2]].copy()
        ro.append(roi)
    if cut_type == 1:    #看切割类型
        '''坐标定位方法'''
        #姓名:
        flag = 0
        (x, y, w, h) = (int(0.95*xx[0]), int(2*yy[0]-yy[1]-hh[1]), int(0.5*ww[0]), int((yy[1]+hh[1]-yy[0])/3))
        if y < 0:
            flag = 1
            y = 0
        roi = image[y:y + h, x:x + w].copy()
        ro.append(roi)
        #性别 + 民族:(不能性别加民族，因为识别会错)
        if flag:
            (x, y, w, h) = (int(0.95*xx[0]), int(yy[0]/3), int(1.5*hh[1]), int(yy[0]/3))
        else:
            (x, y, w, h) = (int(0.95*xx[0]), int(yy[0]-2*(yy[1]+hh[1]-yy[0])/3), int(1.5*hh[1]), int((yy[1]+hh[1]-yy[0])/3))
        roi = image[y:y + h, x:x + w].copy()
        ro.append(roi)
        #民族:
        if flag:
            (x, y, w, h) = (int(xx[0]+0.47*ww[0]), int(yy[0]/3), int(1.5*hh[1]), int(yy[0]/3))
        else:
            (x, y, w, h) = (int(xx[0]+0.47*ww[0]), int(yy[0]-2*(yy[1]+hh[1]-yy[0])/3), int(1.5*hh[1]), int((yy[1]+hh[1]-yy[0])/3))
        roi = image[y:y + h, x:x + w].copy()
        ro.append(roi)
        #年月日  年:int(0.24*ww[0])    # 直接从ID里找，不再识别了
        '''
        if flag:
            (x, y, w, h) = (int(0.95*xx[0]), int(2*yy[0]/3), int(0.9*ww[0]), int(yy[0]/3))
        else:
            (x, y, w, h) = (int(0.95*xx[0]), int(yy[0]-(yy[1]+hh[1]-yy[0])/3), int(0.9*ww[0]), int((yy[1]+hh[1]-yy[0])/3))
        roi = image[y:y + h, x:x + w].copy()
        ro.append(roi)
        '''

    return ro

#----分割部分
class PreprocessCropZeros(object):

    def __init__(self):
        pass

    def do(self, cv2_gray_img):
        height = cv2_gray_img.shape[0]
        width = cv2_gray_img.shape[1]

        v_sum = np.sum(cv2_gray_img, axis=0)
        h_sum = np.sum(cv2_gray_img, axis=1)
        left = 0
        right = width - 1
        top = 0
        low = height - 1

        for i in range(width):
            if v_sum[i] > 0:
                left = i
                break

        for i in range(width - 1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break

        for i in range(height):
            if h_sum[i] > 0:
                top = i
                break

        for i in range(height - 1, -1, -1):
            if h_sum[i] > 0:
                low = i
                break
        if not (top < low and right > left):
            return cv2_gray_img

        return cv2_gray_img[top: low+1, left: right+1]

class PreprocessResizeKeepRatio(object):
    """把图片按原比例缩放至指定大小"""
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def do(self, cv2_img):
        max_width = self.width
        max_height = self.height

        cur_height, cur_width = cv2_img.shape[:2]

        ratio_w = float(max_width)/float(cur_width)
        ratio_h = float(max_height)/float(cur_height)
        ratio = min(ratio_w, ratio_h)

        new_size = (min(int(cur_width*ratio), max_width),
                    min(int(cur_height*ratio), max_height))

        new_size = (max(new_size[0], 1),
                    max(new_size[1], 1),)

        resized_img = cv2.resize(cv2_img, new_size)
        return resized_img

class PreprocessResizeKeepRatioFillBG(object):

    def __init__(self, width, height, fill_bg=False,
                 auto_avoid_fill_bg=True, margin=None):
        self.width = width
        self.height = height
        self.fill_bg = fill_bg
        self.auto_avoid_fill_bg = auto_avoid_fill_bg
        self.margin = margin

    @classmethod
    def is_need_fill_bg(cls, cv2_img, th=0.5, max_val=255):
        image_shape = cv2_img.shape
        height, width = image_shape
        if height * 3 < width:
            return True
        if width * 3 < height:
            return True
        return False

    @classmethod
    def put_img_into_center(cls, img_large, img_small, ):
        width_large = img_large.shape[1]
        height_large = img_large.shape[0]

        width_small = img_small.shape[1]
        height_small = img_small.shape[0]

        if width_large < width_small:
            raise ValueError("width_large <= width_small")
        if height_large < height_small:
            raise ValueError("height_large <= height_small")

        start_width = (width_large - width_small) // 2
        start_height = (height_large - height_small) // 2

        img_large[start_height:start_height + height_small,
                  start_width:start_width + width_small] = img_small
        return img_large

    def do(self, cv2_img):

        if self.margin is not None:
            width_minus_margin = max(2, self.width - self.margin)
            height_minus_margin = max(2, self.height - self.margin)
        else:
            width_minus_margin = self.width
            height_minus_margin = self.height

        cur_height, cur_width = cv2_img.shape[:2]
        if len(cv2_img.shape) > 2:
            pix_dim = cv2_img.shape[2]
        else:
            pix_dim = None

        preprocess_resize_keep_ratio = PreprocessResizeKeepRatio(
            width_minus_margin,
            height_minus_margin)
        resized_cv2_img = preprocess_resize_keep_ratio.do(cv2_img)

        if self.auto_avoid_fill_bg:
            need_fill_bg = self.is_need_fill_bg(cv2_img)
            if not need_fill_bg:
                self.fill_bg = False
            else:
                self.fill_bg = True

        ## should skip horizontal stroke
        if not self.fill_bg:
            ret_img = cv2.resize(resized_cv2_img, (width_minus_margin,
                                                   height_minus_margin))
        else:
            if pix_dim is not None:
                norm_img = np.zeros((height_minus_margin,
                                     width_minus_margin,
                                     pix_dim),
                                    np.uint8)
            else:
                norm_img = np.zeros((height_minus_margin,
                                     width_minus_margin),
                                    np.uint8)
            ret_img = self.put_img_into_center(norm_img, resized_cv2_img)

        if self.margin is not None:
            if pix_dim is not None:
                norm_img = np.zeros((self.height,
                                     self.width,
                                     pix_dim),
                                    np.uint8)
            else:
                norm_img = np.zeros((self.height,
                                     self.width),
                                    np.uint8)
            ret_img = self.put_img_into_center(norm_img, ret_img)
        return ret_img

def extract_peek_ranges_from_array(array_vals, min_range, min_pixs_val=10):
    """设置一下分割纵向单个字时，地址和数字的min_range不同即可，省事一下"""
    start_i = None
    end_i = None
    count = 0
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > min_pixs_val and start_i is None:
            start_i = i
        elif val > min_pixs_val and start_i is not None:
            pass
        elif val < min_pixs_val and start_i is not None:
            end_i = i
            count += 1
            if end_i - start_i >= min_range:
                peek_ranges.append((start_i, end_i))
                count = 0
                start_i = None
                end_i = None
        elif val < min_pixs_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")

    if start_i is not None and i - start_i >= min_range:
        peek_ranges.append((start_i, i))

    return peek_ranges

def _extract_peek_ranges_from_array(array_vals, gap, min_range=13, min_pixs_val=10):
    """传入一个一维array，依次判断元素值是否大于阈值，大于就算到一行里，直到不大于时一行结束，开始下一行
    返回每一行的（开始，结束）元祖的列表"""
    """应该有个这样的逻辑：如果两个文本行间距小于一个阈值，这两个文本行应该属于同一行，也就是字的上下结构间距。
        加一个字符空隙间隔阈值，超过算另一行"""
    """现在的min_range指的是判断本身的长宽是否大于阈值。"""
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > min_pixs_val and start_i is None:    # 一行的开始
            start_i = i
        elif val > min_pixs_val and start_i is not None:
            pass
        elif val < min_pixs_val and start_i is not None:
            if end_i is None and (i - start_i >= min_range):  # 一行的临时结束
                end_i = i
            # 临时结束，下面要继续走，走超出gap范围且仍是空行才能决定刚才是真文本行。
            # 是空行就要置None,不是就要继续走到空行
            elif end_i is not None and (i - end_i > gap):  # 说明end_i是真end点
                peek_ranges.append((start_i, end_i))
                start_i = None
                end_i = None
        elif val < min_pixs_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    """TODO 再加一个逻辑：
    排序统计行或列分割的间隔大小，取前几个间隔的平均作为全局间隔，并合并重叠部分
    (但是这样如何处理年月日区域不同宽度的间隔，分割出生日期时不能执行)"""

    return peek_ranges

def compute_median_w_from_ranges(peek_ranges):
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    return median_w

def median_split_ranges(peek_ranges):
    new_peek_ranges = []
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    for i, peek_range in enumerate(peek_ranges):
        num_char = int(round(widthes[i]/median_w, 0))
        if num_char > 1:
            char_w = float(widthes[i] / num_char)
            for i in range(num_char):
                start_point = peek_range[0] + int(i * char_w)
                end_point = peek_range[0] + int((i + 1) * char_w)
                new_peek_ranges.append((start_point, end_point))
        else:
            new_peek_ranges.append(peek_range)
    return new_peek_ranges

def segment_characters(image, min_range):
    """image：待分割文本图像，每个区域的cv2灰度图，
       min_range: 用以区分纵向切 文字 和 数字 时设的间隔阈值
       return：水平竖直投影分割后的每个文字图像cv2二值图列表
    """
    #cv2_img = cv2.cvtColor(cv2_color_img, cv2.COLOR_RGB2GRAY)
    cv2_img = image
    #height, width = cv2_img.shape
    # 适应性阈值化
    # TODO: 暂定 也可设cv2.threshold 75 数越小阈值能力越强
    adapt_thresh_img = cv2.threshold(cv2_img, 102, 255, cv2.THRESH_BINARY)[1]
    #adapt_thresh_img = cv2.threshold(cv2_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #adapt_thresh_img = cv2.adaptiveThreshold(cv2_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    adapt_thresh_img = 255 - adapt_thresh_img
    cv2_img = 255 - cv2_img
    #cv2.imshow("adap", adapt_thresh_img)
    #cv2.waitKey()
    ## Try to find text lines and chars
    horizontal_sum = np.sum(adapt_thresh_img, axis=1)    # 每一行中的像素值求和
    # 提取出每个行组成下标列表. 每行水平gap设5; 
    # 水平min_range不用改，行宽都差不多，设一个定值。min_pixs_val暂不知为何不同
    peek_ranges = extract_peek_ranges_from_array(horizontal_sum, min_range=20)

    vertical_peek_ranges2d = []
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = adapt_thresh_img[start_y:end_y, :]    # 一行文本

        vertical_sum = np.sum(line_img, axis=0)    # 对每一列求像素值的和，
        # 在纵向切日期区域时，要另设min_range阈值，以正确分出数字1
        vertical_peek_ranges = extract_peek_ranges_from_array(vertical_sum, min_range=min_range, min_pixs_val=30)
        vertical_peek_ranges = median_split_ranges(vertical_peek_ranges)
        vertical_peek_ranges2d.append(vertical_peek_ranges)
    '''## remove noise such as comma
    filtered_vertical_peek_ranges2d = []
    for i, peek_range in enumerate(peek_ranges):
        new_peek_range = []
        median_w = compute_median_w_from_ranges(vertical_peek_ranges2d[i])
        for vertical_range in vertical_peek_ranges2d[i]:
            if vertical_range[1] - vertical_range[0] > median_w*0.5:
                new_peek_range.append(vertical_range)
        filtered_vertical_peek_ranges2d.append(new_peek_range)
    vertical_peek_ranges2d = filtered_vertical_peek_ranges2d
    '''
    norm_width = 32
    norm_height = 32
    char_imgs = []
    #crop_zeros = PreprocessCropZeros()
    resize_keep_ratio = PreprocessResizeKeepRatioFillBG(norm_width, norm_height, fill_bg=False, margin=4)
    for i, peek_range in enumerate(peek_ranges):
        for vertical_range in vertical_peek_ranges2d[i]:
            x = vertical_range[0]
            y = peek_range[0]
            w = vertical_range[1] - x
            h = peek_range[1] - y
            #char_img = adapt_thresh_img[y:y+h+1, x:x+w+1]
            char_img = cv2_img[y:y+h+1, x:x+w+1]
            #char_img = crop_zeros.do(char_img)
            char_img = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            char_img = resize_keep_ratio.do(char_img)
            char_img = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            char_imgs.append(char_img)

    #np_char_imgs = np.asarray(char_imgs)
    np_char_imgs = char_imgs
    return np_char_imgs
#----分割部分结束

def recognition(image, cut_type=1, cls=0):
    '''传入一个cv2灰度图像，返回识别的文字
        TODO: 做成服务，放到内存/显存里随时识别
        按cls区分识别的是全部数据的模型0(地址),还是只有数字的模型1(ID)，
        还是只有性别民族的模型2,还是只有性别的模型3(识别男女)'''
    MOVING_AVERAGE_DECAY = 0.9999
    #characters = []
    '''
    for img in imgs:
        characters.extend(segment_characters(img))
    '''
    character = image
    with tf.Graph().as_default() as g:
        character = cv2.resize(character, (32, 32))
        feed_character = np.array(character)
        character = tf.cast(character, tf.float32)
        character = tf.reshape(character, [32, 32, 1])
        if cls == 0:
            character = tf.image.per_image_standardization(character)

        character = tf.reshape(character, [1,32, 32, 1])
        if cls == 0:
            logits = cnn_model.shijie_inference(character)
        elif cls == 1:
            logits = cnn_model.inference(character)
        elif cls == 2:
            logits = cnn_model.gendernation_inference(character)
        elif cls == 3:
            logits = cnn_model.gender_inference(character)

        logits = tf.nn.softmax(logits)
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        x = tf.placeholder(tf.float32,shape = [32,32])
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            if cls == 0:
                ckpt = tf.train.get_checkpoint_state("./checkpoint")
            elif cls == 1:
                ckpt = tf.train.get_checkpoint_state("./checkpoint_digits")
            elif cls == 2:
                ckpt = tf.train.get_checkpoint_state("./checkpoint_gendernation")
            elif cls == 3:
                ckpt = tf.train.get_checkpoint_state("./checkpoint_gender")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return
            prediction = sess.run(logits, feed_dict={x:feed_character})
            max_index = np.argmax(prediction)

    return max_index


def ocr_main(image, cut_type):
    """直接return一个三个问题和答案的字典"""
    return_json = {"name":"", "gender":"", "nation":"", "birth":"", "address":"", "number":""}
    #image = cv2.imread(image)
    text_areas = find_text(image, cut_type)
    # text_areas: 住址，ID，姓名，性别，民族
    # 人脸切分在find_face里找到然后存入数据库

    # # 对应不同模型，load不同json
    with open("./checkpoint/y_tag.json", 'r') as f:
        chars_dataset = json.load(f)
    with open("./checkpoint_digits/y_tag_digits.json", 'r') as f:
        digits_dataset = json.load(f)
    with open("./checkpoint_gendernation/y_tag_gendernation.json", 'r') as f:
        gendernation_dataset = json.load(f)
    with open("./checkpoint_gender/y_tag_gender.json", 'r') as f:
        gender_dataset = json.load(f)

    seg_chars = segment_characters(text_areas[0], min_range=15)
    for c in seg_chars:
        r = recognition(c)
        return_json["address"]+=chars_dataset[str(r)]

    seg_chars = segment_characters(text_areas[1], min_range=20)
    #print("seg_chars", len(seg_chars))
    i = 0
    for c in seg_chars:
        r = recognition(c, cls=1)
        r = digits_dataset[str(r)]
        if (i == 6 and r == "7") or (i == 12 and r == "7"):
            return_json["number"]+="1"
        else:
            return_json["number"]+=r
        i += 1
    seg_chars = segment_characters(text_areas[2], min_range=30)
    for c in seg_chars:
        r = recognition(c)
        return_json["name"]+=chars_dataset[str(r)]

    seg_chars = segment_characters(text_areas[3], min_range=30)
    '''# 可能会出现两个字，所以只取第一个字
    for c in seg_chars:
        r = recognition(c, cls=3)
        return_json["gender"]+=gender_dataset[str(r)]
    '''
    try:
        r = recognition(seg_chars[0], cls=3)
        return_json["gender"]+=gender_dataset[str(r)]
    except:
        return_json["gender"]+="男"

    seg_chars = segment_characters(text_areas[4], min_range=25)
    for c in seg_chars:
        r = recognition(c, cls=2)
        return_json["nation"]+=gendernation_dataset[str(r)]

    try:
        return_json["birth"] = return_json["number"][6:14]
    except:
        pass

    #return_json = json.dumps(return_json)
    print("return_json", return_json)

    rand = random.sample(range(7), 3)
    question_json = {}
    question = ["你叫什么名字","你的阳历生日是几月几日","你的星座是什么", "你的身份证后6位是多少","你是什么民族的","你的性别是什么","你的生肖是什么"]
    try:
        if return_json["birth"][4] == "0":
            if return_json["birth"][6] == "0":
                ans2 = return_json["birth"][5:6]+"月"+return_json["birth"][7:8]+"日"
            else:
                ans2 = return_json["birth"][5:6]+"月"+return_json["birth"][6:8]+"日"
        else:
            if return_json["birth"][6] == "0":
                ans2 = return_json["birth"][4:6]+"月"+return_json["birth"][7:8]+"日"
            else:
                ans2 = return_json["birth"][4:6]+"月"+return_json["birth"][6:8]+"日"
    except:
        ans2 = "1"
    try:
        if return_json["birth"][4:6] == "12":
            if return_json["birth"][6:8] > "21":
                ans3 = "摩羯座"
            else:
                ans3 = "射手座"
        elif return_json["birth"][4:6] == "11":
            if return_json["birth"][6:8] > "21":
                ans3 = "射手座"
            else:
                ans3 = "天蝎座"
        elif return_json["birth"][4:6] == "01":
            if return_json["birth"][6:8] > "19":
                ans3 = "水瓶座"
            else:
                ans3 = "摩羯座"
        elif return_json["birth"][4:6] == "02":
            if return_json["birth"][6:8] > "18":
                ans3 = "双鱼座"
            else:
                ans3 = "水瓶座"
        elif return_json["birth"][4:6] == "03":
            if return_json["birth"][6:8] > "20":
                ans3 = "白羊座"
            else:
                ans3 = "双鱼座"
        elif return_json["birth"][4:6] == "04":
            if return_json["birth"][6:8] > "19":
                ans3 = "金牛座"
            else:
                ans3 = "白羊座"
        elif return_json["birth"][4:6] == "05":
            if return_json["birth"][6:8] > "20":
                ans3 = "双子座"
            else:
                ans3 = "金牛座"
        elif return_json["birth"][4:6] == "06":
            if return_json["birth"][6:8] > "20":
                ans3 = "巨蟹座"
            else:
                ans3 = "双子座"
        elif return_json["birth"][4:6] == "07":
            if return_json["birth"][6:8] > "21":
                ans3 = "狮子座"
            else:
                ans3 = "巨蟹座"
        elif return_json["birth"][4:6] == "08":
            if return_json["birth"][6:8] > "22":
                ans3 = "处女座"
            else:
                ans3 = "狮子座"
        elif return_json["birth"][4:6] == "09":
            if return_json["birth"][6:8] > "22":
                ans3 = "天秤座"
            else:
                ans3 = "处女座"
        elif return_json["birth"][4:6] == "10":
            if return_json["birth"][6:8] > "22":
                ans3 = "天蝎座"
            else:
                ans3 = "天秤座"
        else:
            ans3 = "金牛座"
    except:
        ans3 = "1"
    try:
        if (2018-int(return_json["birth"][0:4])) % 12 == 0:
            ans7 = "狗"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 1:
            ans7 = "鸡"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 2:
            ans7 = "猴"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 3:
            ans7 = "羊"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 4:
            ans7 = "马"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 5:
            ans7 = "蛇"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 6:
            ans7 = "龙"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 7:
            ans7 = "兔"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 8:
            ans7 = "虎"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 9:
            ans7 = "牛"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 10:
            ans7 = "鼠"
        elif (2018-int(return_json["birth"][0:4])) % 12 == 11:
            ans7 = "猪"
    except:
        ans7 = "1"
    try:
        ans4 = return_json["number"][12:18]
    except:
        ans4 = "1"
    answer = [return_json["name"], ans2, ans3, ans4, return_json["nation"], return_json["gender"], ans7]
    question_json["q1"] = question[rand[0]]
    question_json["q2"] = question[rand[1]]
    question_json["q3"] = question[rand[2]]
    question_json["a1"] = answer[rand[0]]
    question_json["a2"] = answer[rand[1]]
    question_json["a3"] = answer[rand[2]]
    #question_json = json.dumps(question_json)
    return question_json
    
if __name__ == '__main__':
    """
    1. 投影分割
    2. 字符和下标的对应
    2. 依次读单个字符并识别，或者一下子读列表并识别
    """
    '''阈值化参数
    cv2.THRESH_BINARY 	    如果 src(x,y)>threshold ,dst(x,y) = max_value; 否则,dst（x,y）=0
    cv.THRESH_BINARY_INV 	如果 src(x,y)>threshold,dst(x,y) = 0; 否则,dst(x,y) = max_value
    cv.THRESH_TRUNC 	    如果 src(x,y)>threshold，dst(x,y) = max_value; 否则dst(x,y) = src(x,y)
    cv.THRESH_TOZERO 	    如果src(x,y)>threshold，dst(x,y) = src(x,y) ; 否则 dst(x,y) = 0
    cv.THRESH_TOZERO_INV 	如果 src(x,y)>threshold，dst(x,y) = 0 ; 否则dst(x,y) = src(x,y)
    '''
    image = cv2.imread("7.jpg")
    text_areas = find_text(image, 1)
    for i in text_areas:
        cv2.imshow("i", i)
        cv2.waitKey()
    seg_chars = segment_characters(text_areas[3], min_range=30)
    for c in seg_chars:
        cv2.imshow("gender", c)
        cv2.waitKey()

