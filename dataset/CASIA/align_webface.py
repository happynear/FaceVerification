import cv2
import sys
import numpy as np
import os,errno
import random
import shutil
import numpy as np
import math
sys.path.append("D:\\face project\\MTCNN_face_detection_alignment\\code\\codes\\vs\\x64\\Release")
import CascadeFaceDetection

model_folder = "D:/face project/MTCNN_face_detection_alignment/code/codes/MTCNNv2/model/"
#root_path = "E:/datasets/casia-maxpy-clean/CASIA-maxpy-clean"
#root_path = "E:/datasets/lfw"
root_path= "C:/datasets/CASIA-maxpy-clean-aligned-96"

show_debug = False

dst_points = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]])


CascadeCNN = CascadeFaceDetection.CascadeCNN(model_folder + "det1-memory.prototxt", model_folder + "det1.caffemodel",
                     model_folder + "det1-memory-stitch.prototxt", model_folder + "det1.caffemodel",
                     model_folder + "det2-memory.prototxt", model_folder + "det2.caffemodel",
                     model_folder + "det3-memory.prototxt", model_folder + "det3.caffemodel",
                     model_folder + "det4-memory.prototxt", model_folder + "det4.caffemodel",
                     0)  # 0 means to use the first gpu, -1 means to use cpu.


def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.

    If the directory already exists, don't do anything.

    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def open_list(path):
    f_list = open(path, "r")
    all_files = f_list.readlines()
    for line in all_files:
        filename, class_num = line.split()
        align_and_save_face(path, filename)

def AlignWuXiang(input_image, points, output_size = (96, 112), ec_mc_y = 40):
    eye_center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
    mouth_center = ((points[3][0] + points[4][0]) / 2, (points[3][1] + points[4][1]) / 2)
    angle = math.atan2(mouth_center[0] - eye_center[0], mouth_center[1] - eye_center[1]) / math.pi * -180.0
    # angle = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]) / math.pi * 180.0
    scale = ec_mc_y / math.sqrt((mouth_center[0] - eye_center[0])**2 + (mouth_center[1] - eye_center[1])**2)
    center = ((points[0][0] + points[1][0] + points[3][0] + points[4][0]) / 4, (points[0][1] + points[1][1] + points[3][1] + points[4][1]) / 4)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rot_mat[0][2] -= (center[0] - output_size[0] / 2)
    rot_mat[1][2] -= (center[1] - output_size[1] / 2)
    warp_dst = cv2.warpAffine(input_image, rot_mat, output_size)
    return warp_dst


def create_list(path, create_aligned_folder = False):
    f_list = open("{0}/list.txt".format(path), "w")
    class_label = 0
    for parent, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            if create_aligned_folder:
                mkdirP(path + "-aligned"+"/"+dirname)
            for sub_parent, sub_dirnames, sub_filenames in os.walk(path+"/"+dirname):
                for sub_filename in sub_filenames:
                    if(sub_filename.endswith("png") or sub_filename.endswith("jpg") or sub_filename.endswith("bmp")):
                        f_list.write("{0} {1}\r\n".format(dirname+"/"+sub_filename, str(class_label)))
            class_label += 1
    f_list.close()


def align_and_save_face(path, filename):
    img = cv2.imread(os.path.dirname(path) + "/" + filename)
    result = CascadeCNN.Predict(img, 0.3, 80.0)
    if 0 == result.__len__():
        result = CascadeCNN.ForceGetLandmark(img, (55.0, 80.0, 100.0, 120.0))#(50.0, 25.0, 130.0, 170.0)
    default_face = None
    if result.__len__() > 1:
        if show_debug:
            print "got " + str(result.__len__()) + " faces."
        center = (img.shape[1] / 2, img.shape[0] / 2)
        distance_to_center = 99999.0
        for face in result:
            distance = abs(face[0][0] + face[0][2] / 2 - center[0]) + abs(face[0][1] + face[0][3] / 2 - center[1])
            if (distance < distance_to_center):
                distance_to_center = distance
                default_face = face
        if show_debug:
            print default_face[0][3]
            cv2.rectangle(img, (int(default_face[0][0]), int(default_face[0][1])),
                          (int(default_face[0][0] + default_face[0][2]), int(default_face[0][1] + default_face[0][3])),
                          (255, 100, 0), 2)
            for i in range(5):
                cv2.circle(img, (int(default_face[2][i][0]), int(default_face[2][i][1])), 1, (0, 0, 255), 2)
            cv2.imshow("image", img)
            cv2.waitKey(0)
    elif result.__len__() == 1:
        default_face = result[0]
    if default_face is not None:
        #similarTransformation = cv2.estimateRigidTransform(np.array(default_face[2]), dst_points, fullAffine=False)
        alignedImg = AlignWuXiang(img, default_face[2])
        print filename
        #if similarTransformation is not None:
            #print filename + " success."
            #alignedImg = cv2.warpAffine(img, similarTransformation, (96, 112))
        cv2.imwrite(os.path.dirname(path) + "-aligned/" + filename, alignedImg)
        if show_debug:
            cv2.imshow("alignedImg", alignedImg)
            cv2.waitKey(1)
        #else:
        #    print filename + " failed."
        """
        src_points = np.array([[(default_face[2][0][0] + default_face[2][1][0]) / 2, (default_face[2][0][1] + default_face[2][1][1]) / 2],
        [(default_face[2][3][0] + default_face[2][4][0]) / 2, (default_face[2][3][1] + default_face[2][4][1]) / 2]])
        new_dst_points = np.array([[(30.2946 + 65.5318) / 2, (51.6963 + 51.5014) / 2], [(33.5493 + 62.7299) / 2, (92.3655 + 92.2041) / 2]])
        similarTransformation = cv2.estimateRigidTransform(src_points, new_dst_points, fullAffine=False)
        alignedImg = cv2.warpAffine(img, similarTransformation, (96, 112))
        cv2.imshow("image", alignedImg)
        cv2.waitKey(0)
        cv2.imwrite(os.path.dirname(path) + "-aligned/" + filename, alignedImg)
      """
        if show_debug:
            print filename
            cv2.rectangle(img, (int(default_face[0][0]), int(default_face[0][1])),
                          (int(default_face[0][0] + default_face[0][2]),
                           int(default_face[0][1] + default_face[0][3])),
                          (255, 100, 0), 2)
            for i in range(5):
                cv2.circle(img, (int(default_face[2][i][0]), int(default_face[2][i][1])), 1, (0, 0, 255), 2)
            cv2.imshow("image", img)
            cv2.waitKey(0)

def split_train_val(list_path, train_ratio=0.95):
    f_list = open(list_path, "r")
    all_files = f_list.readlines()
    f_train_list = open(os.path.dirname(list_path) + "/" + "train.txt", "w")
    f_val_list = open(os.path.dirname(list_path) + "/" + "val.txt", "w")
    for line in all_files:
        if random.uniform(0, 1) < train_ratio:
            f_train_list.write(line)
        else:
            f_val_list.write(line)

#mkdirP(root_path+"-aligned")
#create_list(root_path, create_aligned_folder=True)
#open_list("{0}/list.txt".format(root_path))
create_list(root_path, create_aligned_folder=False)
split_train_val(root_path + "/list.txt")
#align_and_save_face(root_path+"/list.txt", "Muhammad_Saeed_al-Sahhaf/Muhammad_Saeed_al-Sahhaf_0003.jpg")