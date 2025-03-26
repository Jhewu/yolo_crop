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
MODEL = "yolo12x"
SEED = 42
DATASET = "2018_YOLO"
                           
"""TRAINING"""
EPOCH = 100
MIXED_PRECISION = True
IMG_SIZE = 100 
BATCH = 32

LOAD_AND_TRAIN = False
BEST_MODEL_DIR_TRAIN = "/home/jhewu/main/yolo_crop/runs/detect/train_yolo11n_2025_03_22_22_07_24/weights/best.pt"

"""VALIDATION"""
BEST_MODEL_DIR_VAL = "/home/student/Desktop/YOLO/train_yolo12n_2025_03_12_18_09_25/yolo12n_2018_YOLO/weights/best.pt"

"""PREDICT"""
BEST_MODEL_DIR_PREDICT = "/home/student/Desktop/YOLO/runs/detect/train_yolo12x_2025_03_25_14_03_35/weights/best.pt"
IMAGE_TO_PREDICT = ""