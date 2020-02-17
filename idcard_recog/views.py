from django.shortcuts import render

# Create your views here.
import os
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from idcard_recog import card_ocr as ic
import cv2
import numpy as np

def index(request):
    return HttpResponse(u"欢迎光临 !")
    
def idocr(request):
    re = {}
    print("getidocrpost")
    if request.method == 'POST':
        bin_img = request.body
        image = cv2.imdecode(np.fromstring(bin_img, np.uint8), 1)
        image = np.array(image)
        #re = ic.ocr_main(image, 1)
        userid = request.META.get("HTTP_USERID")
        img_path = os.path.join("images",userid)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        p = os.path.join(img_path,str(userid)+"id.jpg")
        print("IDpath", p)
        cv2.imwrite(p, image)
        re = ic.ocr_main(image, 1)
        print("re:", re)
    #return HttpResponse(u"IDOCR! !")
    return JsonResponse(re)

