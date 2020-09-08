#!/usr/bin/env python
import ncnn
import sys
import cv2
import numpy as np
import time
import sklearn

def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist



start = time.time()
net = ncnn.Net()
param = "model/mobilefacenets.param"
net.load_param(param)
model = "model/mobilefacenets.bin"
net.load_model(model)

imagepath = sys.argv[1]
img = cv2.imread(imagepath)
img_h = img.shape[0]
img_w = img.shape[1]
print(img_h, img_w)
mat_in = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h)
out_mat = ncnn.Mat()
ex = net.create_extractor()
ex.set_light_mode(True)
ex.set_num_threads(4)
ex.input("data", mat_in)
ex.extract("fc1", out_mat)
mat_np = np.array(out_mat)
print(time.time()-start)
print(mat_np)
print(mat_np.shape)
 


start = time.time()
imagepath = sys.argv[2]
img = cv2.imread(imagepath)
img_h = img.shape[0]
img_w = img.shape[1]
print(img_h, img_w)
mat_in = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h)
out_mat = ncnn.Mat()
ex = net.create_extractor()
ex.set_light_mode(True)
ex.set_num_threads(4)
ex.input("data", mat_in)
ex.extract("fc1", out_mat)
mat_np = np.array(out_mat)
print(time.time()-start)
print(mat_np)
print(mat_np.shape)
 


