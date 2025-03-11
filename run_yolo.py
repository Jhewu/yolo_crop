from ultralytics import YOLO
import torch 
import os
import csv
import time

# Change parameters here
from parameters import *

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def GetCurrentTime(): 
    current_time = time.localtime()
    return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

""""Main Runtime"""
def RunYOLO(mode):
    if mode == "train":
        print("/\nStarting training...")
       
        if LOAD_AND_TRAIN: 
            model = YOLO(BEST_MODEL_DIR_TRAIN)
        else: 
            # Load pretrained model (recommended for training)
            model = YOLO(f"{MODEL}.pt")

        # Train the model
        results = model.train(data=f"./datasets/{DATASET}.yaml", 
                              epochs=EPOCH, 
                              seed=SEED, 
                              batch=-1,
                              plots=True,
                              project=f"{MODE}_{MODEL}_{GetCurrentTime()}",
                              name=f"{MODEL}_{DATASET}")
        print(f"\nFinished training, please check your directory for folder named 'train-....")
        
    elif mode == "val":
        print("/\nStarting validation...\n")
        print(f"Fetching weights from...{BEST_MODEL_DIR_VAL}\n")
        model = YOLO(BEST_MODEL_DIR_VAL)
        metrics = model.val(plots=True, 
                            name=f"{MODE}_{MODEL}_{DATASET}")
        print(f"\nmAP50-95: {metrics.seg.map}\n")
        print(f"\nFinished validation, please check your directory for folder named 'val-....")
    
    elif mode == "predict":
        print("\nStarting prediction...\n")
        print(f"Fetching weights from...{BEST_MODEL_DIR_PREDICT}\n")
        model = YOLO(BEST_MODEL_DIR_PREDICT)

        results = model(IMAGE_TO_PREDICT)
        
        # Save the prediction
        for result in results:
            result.save(filename="result.jpg")  # save to disk

        print(f"\nFinished prediction, please check your directory for a file named 'results.jpg'")

if __name__ == "__main__":
    RunYOLO(MODE)
