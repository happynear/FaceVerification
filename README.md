# FaceVerification
A messy code for developing a face verfication program. 

It includes a C++ face detection / alignment program, [joint bayesian](http://home.ustc.edu.cn/~chendong/JointBayesian/) and several supplementary codes. Feel free to use the codes if you need.

If you are also interested in face verification, please contact me via the issue.

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
    
    Accuracy on training set is about 89.5%~91.5%. LFW result with L2 or cosine has reached what the paper claimed. Joint Bayesian seems to be strongly overfit.
    
2. Training DeepID2 (siamese network)

    create database. done.
