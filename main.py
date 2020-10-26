import os
import sys
import time
import pickle
import cv2
import numpy as np
import mbfn
import retinaface

if __name__ == "__main__":
    allf = pickle.load(open("features", 'rb'))
    print("database has:", len(allf), "persons")

    if len(sys.argv) != 2:
        print("Usage: %s [imagepath]\n"%(sys.argv[0]))
        sys.exit(0)
   
    mfn = mbfn.MobileFaceNetV3()
    s = time.time()

    imagepath = sys.argv[1]
    m = cv2.imread(imagepath)
    net =  retinaface.RetinaFace()
    faceobjects = net(m)
    lm = [[p.x,p.y] for p in faceobjects[0].landmark]
    lm = np.array(lm)
    features_a = mfn.extract(m, lm)
    print("used", time.time()-s, "s")
    most = 0
    filename = ""
    for x in allf:
        features_b = x['features']
        similar = mbfn.return_similarity(features_a, features_b)
        print("with: ", x['file'], similar)
        if similar > most:
            most = similar
            filename = x['file']
    print("The most similar is:", filename, most)
