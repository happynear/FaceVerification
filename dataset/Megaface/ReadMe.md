# Matlab codes for evaluation on MegaFace

Requirement: 
1. [MatMTCNN](https://github.com/happynear/MTCNN_face_detection_alignment/tree/master/code/codes/vs/MatCascadeFaceDetection). 
If you are not using Windows, you need to modify some of the codes to directly use the [Matlab version of MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment). If you have done such a work, I'm glad to merge your codes.

2. [Megaface](http://megaface.cs.washington.edu/participate/challenge.html). Please download 
`Our dataset`, `FaceScrub full tgz`, `FaceScrub bounding boxes actors txt`,
`FaceScrub bounding boxes actresses txt`, `Linux Development Kit`. 

Procedure:
1. Align Facescrub: `align_facescrub.m`. Then mannully confirm some failed samples through `align_facescrub_failures.m`.

2. Align Megaface: `align_megaface_from_list.m`. Then align failed samples through `align_megaface_failures.m`.

3. Extract Features using `extract_facescrub_feature.m`, `extract_megaface_feature.m`.


Alignment Logic:

The provided 3-point labels are the most accurate. The second accurate information is the bounding box. All 5-point labels are totally wrong.
So our logic is as follows.

1. `align_megaface_from_list.m`: If there is 3-point label, rotate and crop the image according to it. Then detect and align face from the cropped image.
2. `align_megaface_failures.m`: If there is no 3-point label or failed to detect from the cropped image, detect face from the raw image.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If the detected face and the given bounding box's IoU is over 30%, align this face. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If the IoU is below 30%, use the last two networks of MTCNN to forcely get the face score and 5 keypoints from the cropped image based on the given bounding box. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If the face score is above 0.3, use the detected 5 points to align the face. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If all methods are failed, directly crop the middle area of a face as the aligned face.
