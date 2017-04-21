# FaceVerification
A messy code for developing a face verfication program. 

It includes a C++ face detection / alignment program, [joint bayesian](http://home.ustc.edu.cn/~chendong/JointBayesian/) and several supplementary codes. My Caffe model define file is also provided. Note that I use a fresh layer called Insanity to replace the ReLU activation. The Insanity layer can be found in [my Caffe repository](https://github.com/happynear/caffe-windows). Please feel free to use the codes if you need.

If you are also interested in face verification, please contact me via the issue.

The [CASIA-webface dataset](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) is really very dirty, and I believe that if someone could wash it up, the accuracy would increase further. If you did so, please kindly contact me. I will pay for it.

**Good News:** [@潘泳苹果皮](http://weibo.com/maxpanyong) and his colleagues have washed the CASIA-webface database manually. After washing, 27703 wrong images are deleted. The washed list can be downloaded from http://pan.baidu.com/s/1hrKpbm8 . Great thanks to them!

Update
==========

2017/04/21 The project page of my new paper, NormFace: L2 HyperSphere Embedding for Face Verification, is created on https://github.com/happynear/NormFace. Using the new loss functions to fine-tune a network, the accuracy will increase a little bit higher.

2017/02/18 I trained a [center-face](https://github.com/ydwen/caffe-face) model on [MS-Celeb-1M dataset](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) and get 99.3% on LFW. Here is the model (http://pan.baidu.com/s/1jIJT4Rc) and the aligned LFW images (http://pan.baidu.com/s/1bp7qzJh). To run the evaluation, you need to load LFW pairs by `getPairs.m` in `aligned_lfw.zip`, extract feature by `ReadFeatureLFW.m` and get the accuracy by `lfwPCA.m`. The function `pcaApply` used in `lfwPCA.m` is from [pdollar-toolbox](https://github.com/pdollar/toolbox).

**Recently I talked with Yandong Wen. He said that there were more than 1,000 identities' overlap between MS-Celeb-1M and LFW. So this accuracy, 99.3%, is not a reliable value, and the performance on other datasets may not be very good.**

2015/07/05 Added a matlab face alignment wrapper (MatAlignment.cpp). Now you can do the face alignment job in matlab. A demo (VerificationDemo.m) is also privided. 

Progress
===========
1. Training DeepID (pure softmax network).

    create database. done.
    
    iteration 360,000, lr=0.01,
    
        lfw verification: L2 : 95.9%, jb : 
    
    iteration 500,000, lr=0.001,
    
        lfw verification: L2 : 96.8%, jb : 93.3% (strongly overfit, it's >99% for lfw training set).
        
    iteration 660,000, lr=0.0001,
    
        lfw verification: L2 : 96.78% (converged)
    
    Accuracy on training set is about 89.5%~91.5%. LFW result with L2 or cosine has reached what the paper claimed. Joint Bayesian seems to be strongly overfit. The main reason is that I only train Joint Bayesian on the lfw training set, not CASIA-WebFace. Joint Bayesian for over 10,000 classes is too costy for my PC. 
    
    This model is public available now: http://pan.baidu.com/s/1qXhNOZE . Mean file is in http://pan.baidu.com/s/1eQYEJU6 .
    
    Another model with resolution of 64*64 is trained. By ensembling the two models, accuracy increases to 97.18%.
    
2. Training DeepID2 (siamese network)

    create database. done.
    
    My GPU memory is only 2GB. It is too slow to train a siamese network. I need Titan X!!!
