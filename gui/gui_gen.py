#!/usr/bin/env python3
#coding:utf8
import sys 
sys.path.append("..") 
import PySimpleGUI as sg
from PIL import Image, ImageDraw, ImageTk, ImageFont
import cv2
import numpy as np  # 数据处理的库 Numpy
import os           # 读写文件
import time
import pickle
from retinaface import RetinaFace
from mbfn import MobileFaceNetV3
import random
import string


path_photos_from_camera = "photos/"

def RandomStr(num=6):
    return ''.join(random.sample(string.ascii_letters + string.digits, num))

def pre_work_mkdir():
    if os.path.isdir(path_photos_from_camera):
        pass
    else:
        os.mkdir(path_photos_from_camera)


def start():
    mfn = MobileFaceNetV3()
    pre_work_mkdir()
    all_features  = None
    with open(".features", 'rb') as db:
        all_features  = pickle.load(db)

    font = ImageFont.truetype(font='simsun.ttc', size=40)

    taken = False
    img_rd = None
    layout= [
            [sg.Text(' 请输入姓名:', size=(10,1)), sg.InputText(key="name", default_text="张三", size=(90,1), pad=(10,0))],
            [sg.Button("拍照", pad=(1,1)), sg.Button('录入',pad=(1,1)), sg.Button('取消',pad=(1,1))], 
            [sg.Image(filename="logo.png", size=(1216,912), pad=(11,11), key="image")],
        ]

    sg.theme('DarkTeal10')
    window = sg.Window('人脸注册', layout, font="Any 20", resizable=True, location=(250, 15), size=(1250, 1050), finalize=True)
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FPS, 12);

    while True:
        event, values = window.read(1)
        if event == sg.WIN_CLOSED:
            capture.release()
            break
        elif  event == '取消':
            taken = False
        elif  event == '录入':
            if len(values['name']) <= 0:
                sg.PopupOK("姓名不能为空", auto_close=True, font="Any 18", auto_close_duration=1)
            elif taken == False:
                sg.PopupOK("   先拍照", auto_close=True, font="Any 18", auto_close_duration=1)
            else:
                sg.PopupOK("   请稍等", auto_close=True, font="Any 18", auto_close_duration=1)
                faces = Detect(img_rd)
                if len(faces) >= 1:
                    if len(faces) >= 2:
                        sg.PopupError("每次仅能一人注册", font="Any 18", auto_close=True, auto_close_duration=2)
                        taken = False
                        continue
                    for obj in faces:
                        print(obj.rect.x, obj.rect.y, obj.rect.w, obj.rect.h)
                        x = int(obj.rect.x)
                        y = int(obj.rect.y)
                        height = int(obj.rect.h)
                        width = int(obj.rect.w)
                        h = int(height/2)
                        w = int(width/2)
                        print(x+w, y+h, x-w, y-h)
                        if (x + w) > 640 or (y + h > 480) or (x - w < 0) or (y - h < 0):
                            sg.PopupError("特征提取失败，确保人脸在红色方框内", font="Any 18", auto_close=True, auto_close_duration=2)
                        else:
                            img_blank = np.zeros(((height*2), (width*2), 3), np.uint8)
                            for ii in range(height*2):
                                for jj in range(width*2):
                                    img_blank[ii][jj] = img_rd[y - h + ii][x - w + jj]
                            name = RandomStr()
                            file_name = path_photos_from_camera + "/{}.jpg".format(name)
                            cv2.imwrite(file_name, img_blank)
                            lm = [[p.x,p.y] for p in obj.landmark]
                            lm = np.array(lm)
                            features = mfn.extract(img_blank, lm)
                            all_features.append({"file":file_name, "features":features, "name":values['name']})
                            if len(features) != 128:
                                person_cnt -= 1
                                os.remove(file_name)
                                print("gen feature error, so remove jpg")
                                sg.PopupError("特征提取失败!", font="Any 18", auto_close=True, auto_close_duration=2)
                            else:
                                print("Save into：", file_name)
                                pickle.dump(all_features, open('.features','ab+'))
                                sg.PopupOK("人脸注册成功", font="Any 18", auto_close=True, auto_close_duration=2)
                                #只录入一次就退出
                                event, values = window.read(3000)
                                capture.release()
                                window.close()
                                return
                else:
                    sg.PopupError("未检测到人脸，确保人脸在红色方框内", font="Any 18", auto_close=True, auto_close_duration=2)
                taken = False
        elif  event == '拍照':
            taken = True

        if taken == False:
            ret, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            draw = ImageDraw.Draw(im)
            draw.text(xy=(130, 400), text='人脸铺满方框保持静止', fill=(255, 0, 0), font=font)
            draw.rectangle(xy=(200, 80, 440, 340), fill=None, outline="red", width=2)
            im = im.resize((1216, 912),Image.ANTIALIAS)
            tkimage = ImageTk.PhotoImage(image=im)
            window.FindElement("image").update(data=tkimage)
            img_rd = frame

    window.close()
                 
def Detect(img):
    net =  RetinaFace()
    faceobjects = net(img)
    return faceobjects

if __name__ == "__main__":
    start()
