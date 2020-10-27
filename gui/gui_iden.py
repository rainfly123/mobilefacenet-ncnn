#!/usr/bin/env python3
#coding:utf8
import PySimpleGUI as sg
from PIL import Image, ImageDraw, ImageTk, ImageFont
import numpy as np   
import cv2 
import os
import time
import subprocess
from retinaface import RetinaFace
from mbfn import MobileFaceNetV3
from mbfn import return_similarity
#from sfas import fake

def show_succeed(name, filename):
    img = Image.open(filename).resize((350,300))
    frame = ImageTk.PhotoImage(img)
    lyout= [
            [sg.Image(filename=None, size=(350,300), pad=(10,10), key="image"),
            sg.Text("姓名:"), sg.Text(name)],
           ]
    win = sg.Window('识别成功', lyout, font="Any 20", location=(250, 700), size=(1250, 300),  finalize=True)
    win.FindElement("image").update(data=frame)
    event, values = win.read(2000)
    win.close()

def start():
    mfn = MobileFaceNetV3()
    layout= [
            [sg.Image(filename="logo.png", size=(1200,950), pad=(25,25), key="image")],
            ]
    last_time = time.time()
    cap= cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 12);

    allf = pickle.load(open(".features", 'rb'))
    print("database has:", len(allf), "persons")

    process_image = True
    sg.theme('DarkTeal10')   # Add a touch of color
    window = sg.Window('人脸识别', layout, font="Any 20", location=(250, 15), size=(1250, 1020), resizable=True, finalize=True)
    font = ImageFont.truetype(font='simsun.ttc', size=40)

    while True:
        event, values = window.read(1)
        if event == sg.WIN_CLOSED:
            cap.release()
            break

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)
        draw.text(xy=(130, 40), text='确保人脸在红色方框内', fill=(255, 255, 0), font=font)
        draw.rectangle(xy=(150, 130, 500, 400), fill=None, outline="red", width=2)
        im = im.resize((1200, 950),Image.ANTIALIAS)
        tkimage = ImageTk.PhotoImage(image=im)
        window.FindElement("image").update(data=tkimage)

        faces = []
        if (time.time() - last_time > 2.0) and process_image:
            img_rd = cv2.resize(frame, (320, 240))
            faces = Detect(img_rd)
        process_image = not process_image
        if len(faces) != 0:
            for obj in faces:
                lm = [[p.x,p.y] for p in obj.landmark]
                lm = np.array(lm)
                features_a = mfn.extract(m, lm)
                most = 0.0
                most_p = None
                for x in allf:
                    features_b = x['features']
                    similar = mbfn.return_similarity(features_a, features_b)
                    if similar > most:
                        most = similar
                        most_p = x
                if most_p and most > 0.8:
                    #val = fake.real_face_ncnn(img_rd, [obj.rect.x, obj.rect.y, obj.rect.x+obj.rect.w, obj.rect.y+obj.rect.h])
                    #if val <= 0.95:
                    #    break
                    show_succeed(x['name'], x['file'])

                    im = Image.fromarray(img_rd)
                    draw = ImageDraw.Draw(im)
                    for i in range(68):
                        x,y = shape.part(i).x, shape.part(i).y
                        draw.ellipse((x,y, x+1, y+1), 'red')
                    im = im.resize((1200, 950),Image.ANTIALIAS)
                    tkimage = ImageTk.PhotoImage(image=im)
                    window.FindElement("image").update(data=tkimage)
                    #只录入一次就退出
                    event, values = window.read(2000)
                    sg.Popup(" 柜门已打开", title="友情提示", font="Any 18", custom_text="  确认  ")
                    #ff = subprocess.Popen(["ffplay", "-fs", "-autoexit",'abc.mp4'])
                    #ff.wait()
                    cap.release()
                    window.close()
                    return
                    last_time = time.time()
    window.close()

def Detect(img):
    net =  RetinaFace()
    faceobjects = net(img)
    return faceobjects

if __name__ == "__main__":
    start()