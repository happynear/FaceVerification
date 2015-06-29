# FaceVerification
A messy code for developing a face verfication program. 

It includes a C++ face detection / alignment program, [joint bayesian](http://home.ustc.edu.cn/~chendong/JointBayesian/) and several supplementary codes. My Caffe model define file is also provided. Note that I use a fresh layer called Insanity to replace the ReLU activation. The Insanity layer can be found in [my Caffe repository](https://github.com/happynear/caffe-windows). Please feel free to use the codes if you need.

If you are also interested in face verification, please contact me via the issue.

The [CASIA-webface dataset](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) is really very dirty, and I believe that if someone could wash it up, the accuracy would increase further. If you did so, please kindly contact me. I will pay for it.

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
    
    Another model with resolution of 64*64 is trained. By ensembling the two models, accuracy increases to 97.18%.
    
2. Training DeepID2 (siamese network)

    create database. done.
    
    My GPU memory is only 2GB. It is too slow to train a siamese network. I need Titan X!!!
