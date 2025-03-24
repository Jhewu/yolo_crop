"""
COMMENT: All hyperparameters configured within
this file. The other hyperparameters
are in "default" mode, check 
https://docs.ultralytics.com/usage/cfg/#tasks
for more information...
"""

"""GENERAL"""
# OPTIONS: train, val, predict
MODE = "train"           
MODEL = "yolo11n"
SEED = 42
DATASET = "2018_YOLO"
                           
"""TRAINING"""
EPOCH = 22
MIXED_PRECISION = True
IMG_SIZE = 100

LOAD_AND_TRAIN = True
BEST_MODEL_DIR_TRAIN = "/home/jhewu/main/yolo_crop/runs/detect/train_yolo11n_2025_03_22_22_07_24/weights/best.pt"

"""VALIDATION"""
BEST_MODEL_DIR_VAL = "/home/student/Desktop/YOLO/train_yolo12n_2025_03_12_18_09_25/yolo12n_2018_YOLO/weights/best.pt"

"""PREDICT"""
BEST_MODEL_DIR_PREDICT = ""
IMAGE_TO_PREDICT = ""