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
            faces = detector(img_rd, 0)
        process_image = not process_image
        if len(faces) != 0:
            for k in range(len(faces)):
                shape = predictor(img_rd, faces[k])
                features_cap_arr = facerec.compute_face_descriptor(img_rd, shape)
                e_distance_list = []
                for i in range(len(features_known_arr)):
                    if str(features_known_arr[i][0]) != '0.0':
                        e_distance_tmp = return_euclidean_distance(features_cap_arr, features_known_arr[i])
                        print("with person", str(i + 1), "the e distance: ", e_distance_tmp)
                        e_distance_list.append(e_distance_tmp)
                    else:
                        e_distance_list.append(999)

                if e_distance_list and min(e_distance_list) < 0.42:
                    val = fake.real_face_ncnn(img_rd, [faces[k].left(), faces[k].top(), faces[k].right(), faces[k].bottom()])
                    if val <= 0.95:
                        break
                    similar_person_num = e_distance_list.index(min(e_distance_list))
                    show_succeed(similar_person_num + 1)

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

if __name__ == "__main__":
    start()
