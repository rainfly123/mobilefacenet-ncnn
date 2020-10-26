#!/usr/bin/env python
import ncnn
import sys
import cv2
import numpy as np
import time
import sklearn
import preprocess
from PIL import Image

class Mtcnn():
    def __init__(self):
        self.net = ncnn.Net()
        self.pparam = "model/det1.param"
        self.pmodel = "model/det1.bin"
        self.rparam = "model/det2.param"
        self.rmodel = "model/det2.bin"
        self.oparam = "model/det3.param"
        self.omodel = "model/det3.bin"
        self.nparam = "model/det4.param"
        self.nmodel = "model/det4.bin"
        self.pnet.load_param(self.pparam)
        self.pnet.load_model(self.pmodel)
        self.rnet.load_param(self.rparam)
        self.rnet.load_param(self.rparam)
        self.onet.load_param(self.oparam)
        self.onet.load_model(self.omodel)
        self.nnet.load_model(self.nmodel)
        self.nnet.load_model(self.nmodel)

    def extract(self, m, l , img_file, bbox=None, landmark=None):
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        #img_aligned = preprocess.preprocess(img, bbox, landmark, image_size="112,112")
        cropped_face = Image.fromarray(img)
        test_face = resize(cropped_face)
        test_face = to_tensor(test_face)
        test_face = normalize(test_face)
        test_face.unsqueeze_(0)
        """
        _mean_val = [103.94, 116.78, 123.68]
        _norm_val = [0.017, 0.017, 0.017]
        mat_in = ncnn.Mat.from_pixels(img_aligned, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 112,112)
        mat_in.substract_mean_normalize(_mean_val, _norm_val);
        out_mat = ncnn.Mat()
        ex = self.net.create_extractor()
        ex.set_light_mode(True)
        ex.set_num_threads(4)
        ex.input("data", mat_in)
        ex.extract("fc1", out_mat)
        mat_np = np.array(out_mat)
        """
        temp = m(torch.Tensor(test_face))
        mat_t = l(temp)
        t = mat_t.view(-1)
        mat_np = t.detach().numpy()
        out=""
        for x in range(128):
            t = "{},".format(mat_np[x])
            out += t
        out = out[:-1] + '\n'
        with open('feature', "a") as fs:
            fs.write(out)
        return mat_np
 
if __name__ == '__main__':
    import glob
    import timm
    import torch
    m = timm.create_model('resnet50', pretrained=True, num_classes=0)
    l = torch.nn.Linear(2048,128)
   # f = glob.glob('data/*.jpg')
    a = MobileFaceNetV3()
    i = 1
    #for image in f:
    if 1:
        img = cv2.imread("rain.jpg", cv2.IMREAD_COLOR)
        total, loc = landmarks.landmarks(img)
        one = a.extract(m, l, img, None, np.array(loc[0]))
        i+=1
    if 1:
        img = cv2.imread("abc.jpg", cv2.IMREAD_COLOR)
        total, loc = landmarks.landmarks(img)
        t = time.time()
        two = a.extract(m, l, img, None, np.array(loc[0]))
        print(time.time()-t)
        i+=1
    print(return_euclidean_distance(one,two))

    
