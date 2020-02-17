from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def face_verification(request):
    return HttpResponse("face_verification!")
