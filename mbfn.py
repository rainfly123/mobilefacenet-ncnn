#!/usr/bin/env python
import ncnn
import sys
import cv2
import numpy as np
import time
import sklearn
import preprocess
import dlib
import landmarks

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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
        img = None
        if isinstance (img_file, str):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        else:
            img = img_file

        """
        faces = detector(img, 0)
        if len(faces) != 0:
            for k in range(len(faces)):
                shape = predictor(img, faces[k])
                cv2.circle(img, (faces[0].left(), faces[0].top()), 3, (0,255,0), 0)
                cv2.circle(img, (faces[0].right(), faces[0].bottom()), 3, (0,255,0), 0)
                print(faces[0])
                for i in range(68):
                    x,y = shape.part(i).x, shape.part(i).y
                    print(x,y)
                    cv2.circle(img, (x, y), 3, (0,0,255), 0)
                cv2.imshow("d", img)
                cv2.waitKey(0)
        return []
        """

        img_h = img.shape[0]
        img_w = img.shape[1]
        img_aligned = preprocess.preprocess(img, bbox, landmark, image_size="112,112")
        _mean_val = [103.94, 116.78, 123.68]
        _norm_val = [0.017, 0.017, 0.017]
        mat_in = ncnn.Mat.from_pixels(img_aligned, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 112, 112)
        mat_in.substract_mean_normalize(_mean_val, _norm_val);
        out_mat = ncnn.Mat()
        ex = self.net.create_extractor()
        ex.set_light_mode(True)
        ex.set_num_threads(4)
        ex.input("data", mat_in)
        ex.extract("fc1", out_mat)
        mat_np = np.array(out_mat)
        #print(mat_np)
        return mat_np
 
if __name__ == '__main__':
    a = MobileFaceNetV3()
    t = time.time()
    imagepath = sys.argv[1]
    img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
    total, loc = landmarks.landmarks(img)
    for x in range(total):
        one = a.extract(img, None, np.array(loc[x]))

    imagepath = sys.argv[2]
    img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
    total, loc = landmarks.landmarks(img)
    for x in range(total):
        two = a.extract(img, None, np.array(loc[x]))
    print(time.time()-t)
    print(return_euclidean_distance(one, two) )
