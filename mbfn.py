#!/usr/bin/env python
import ncnn
import sys
import cv2
import numpy as np
import time
import sklearn
import preprocess

def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist

class MobileFaceNetV3():
    def __init__(self):
        self.net = ncnn.Net()
        self.param = "model/mobilefacenets.param"
        self.model = "model/mobilefacenets.bin"
        self.net.load_param(self.param)
        self.net.load_model(self.model)

    def extract(self, img_file, bbox=None, landmark=None):
        start = time.time()
        img = None
        if isinstance (img_file, str):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        else:
            img = img_file
        img_h = img.shape[0]
        img_w = img.shape[1]
        print(img_h, img_w)
        img_aligned = preprocess.preprocess(img, bbox, landmark, image_size="112,112")
        mat_in = ncnn.Mat.from_pixels(img_aligned, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h)
        out_mat = ncnn.Mat()
        ex = self.net.create_extractor()
        ex.set_light_mode(True)
        ex.set_num_threads(4)
        ex.input("data", mat_in)
        ex.extract("fc1", out_mat)
        mat_np = np.array(out_mat)
        print(time.time()-start)
        print(mat_np)
        print(mat_np.shape)
 
if __name__ == '__main__':
    imagepath = sys.argv[1]
    a = MobileFaceNetV3()
    a.extract(imagepath)
