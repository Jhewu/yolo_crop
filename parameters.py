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
SEED = 42
MODEL = "yolo12n"

DATASET = "2018_YOLO"
                           
"""TRAINING"""
EPOCH = 100

LOAD_AND_TRAIN = False
BEST_MODEL_DIR_TRAIN = ""

"""VALIDATION"""
BEST_MODEL_DIR_VAL = ""

"""PREDICT"""
BEST_MODEL_DIR_PREDICT = ""
IMAGE_TO_PREDICT = ""