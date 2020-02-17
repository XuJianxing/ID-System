from django.shortcuts import render

# Create your views here.
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from face_anti_spoofing import real_time as fr
from face_verification import compare as fc
import cv2
import numpy as np
from scipy import misc
import tensorflow as tf
import copy
from face_verification import facenet
from face_verification import align
from face_verification.align import detect_face

def faceantitest(request):
    return HttpResponse("face_anti_spoofing!")


class now_datas():
    pass
now = now_datas()
now.spoof_now_id = "0"
now.count_enough = 0
now.spoofing_count = 0
print("now:", now.__dict__)

def faceantispoofing(request):
    data = {}  # 3个字段: 
    if request.method == "POST":
        bin_img = request.body
        image = cv2.imdecode(np.fromstring(bin_img, np.uint8), 1)
        image = np.array(image)    # 记录下现在的array图像数据
        #re = ic.ocr_main(image, 1)
        userid = request.META.get("HTTP_USERID")  # 现在的用户ID
        if userid != now.spoof_now_id:
            now.spoof_now_id = userid
            now.spoofing_count = 1
        else:
            now.spoofing_count += 1
        img_path = os.path.join("images",userid)  # 保存图片的文件夹的路径
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        path = os.path.join(img_path,
            str(userid)+str(now.spoofing_count)+"spoof.jpg")
        print("spoofimgpath", path)    # 保存图片的最终的路径和文件名
        print(image.shape)
        cv2.imwrite(path, image)  # 保存图片
        print(userid)
        data["spoof_result"] = fr.face_anti(image)

        # ID里的face图（比对时自带人脸检测，所以不用切出人脸了
        p1 = os.path.join(img_path,str(userid)+"id.jpg")
        print("p1 path", p1)
        # 和现在的图做比较，保存比较结果（0，1）
        data["verification_result"] = fc.compare_main(p1, path)
        if now.spoofing_count == 3:
            data["count_enough"] = 1
        print('data', data)
    return JsonResponse(data)
