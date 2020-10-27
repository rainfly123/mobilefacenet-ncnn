# mobilefacenet-ncnn
this software is used to face recognition   
it based on [pyncnn](https://github.com/caishanli/pyncnn)
pyncnn is a python wrapper of [ncnn](https://github.com/Tencent/ncnn) with [pybind11](https://github.com/pybind/pybind11), only support python3.x now.   



it use the [retinaface](https://github.com/deepinsight/insightface/tree/master/RetinaFace) for detect face   
and use mobilefacenetv3 for generate 128 dimension face feature vector 

My weichat id: Rainfly003   
It would be appreciated if you buy me a cup of coffee.

1. gen.py used to generate face features file,   
it scans all the images in the dirctory which you input as a command parameter   
and generate featues file use pickle module

2. main.py used to find the images that have the most 10 highest similarity  with the image you input  

3. mbfn.py is the mobile face net Class

4. retiface.py is used to detect face region and 5 points (left eye, right eye , nose, left/right mouth corner)  


# IF you want to test it  in GUI mode   
please install PySimpleGui, numpy,  PIL and opencv  at first   
then   

```
git clone https://github.com/rainfly123/mobilefacenet-ncnn/
cd mobilefacenet-ncnn
cd gui
git submodule update --init --recursive
```

the sfas module is used for Silent-Face-Anti-Spoofing   
