# FaceVerification
A messy code for developing a face verfication program. 

It includes a C++ face detection / alignment program, joint bayesian and several supplementary codes. Please just use the codes if you need.

If you are also interested in face verification, please contact me via the issue.

Progress
===========
1. Training DeepID (pure softmax network).

    create database. done.
    
    iteration 360,000, lr=0.01,
    
        lfw verification: l2 : 95.9%, jb : 
    
    iteration 500,000, lr=0.001,
    
        lfw verification: l2 : 96.8%, jb : 93.3% (strongly overfit, it's >99% for lfw training set).
    
2. Training DeepID2 (siamese network)

    create database. done.
