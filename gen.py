#!/usr/bin/env python
import ncnn
import sys
import cv2
import numpy as np
import time
import sklearn
import preprocess


def return_similarity(feature_1, feature_2):
    #feature_1 = np.array(feature_1)
    #feature_2 = np.array(feature_2)
    return np.sum(feature_1 * feature_2)

class MobileFaceNetV3():
    def __init__(self):
        self.net = ncnn.Net()
        self.param = "model/mobilefacenets.param"
        self.model = "model/mobilefacenets.bin"
        self.net.load_param(self.param)
        self.net.load_model(self.model)
        self.num_threads = 2

    def extract(self, img_file, landmark=None):
        img = None
        if isinstance (img_file, str):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        else:
            img = img_file

        img_h = img.shape[0]
        img_w = img.shape[1]
        img_aligned = preprocess.preprocess(img, landmark, image_size="112,112")
        #img_aligned = img_aligned.unsqueeze(0)
        mat_in = ncnn.Mat.from_pixels(img_aligned, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 112, 112)
        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)
        ex.input("data", mat_in)
        out = ncnn.Mat()
        ex.extract("fc1", out)
        out = np.array(out)
        out = np.divide(out, np.sqrt(np.sum(np.square(out))))
        print(out)
        return out
 

def all_files(path):
    allfiles = list()
    for root, dirs, files in os.walk(path):
        for name in files:
            allfiles.append(os.path.join(root, name))
        for dirn in dirs:
            all_files(os.path.join(root, dirn))

    return allfiles

if __name__ == '__main__':
    import retinaface
    import pickle
    import os

    if len(sys.argv) != 2:
        print("Usage: %s [imagepath]\n"%(sys.argv[0]))
        sys.exit(0)
   
    mfn = MobileFaceNetV3()
    s = time.time()

    imagepath = sys.argv[1]
    allf = all_files("/home/rain/lfw")
    all_features = list()
    total = 100000
    for x in allf:
        m = cv2.imread(x)
        net =  retinaface.RetinaFace()
        faceobjects = net(m)
        lm = [[p.x,p.y] for p in faceobjects[0].landmark]
        lm = np.array(lm)
        features = mfn.extract(m, lm)
        all_features.append({"file":x, "features":features})
        total -= 1
        if total < 0:
            break

    pickle.dump(all_features, open('features','ab+'))
