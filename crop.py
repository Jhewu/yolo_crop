"""
This code is for YOLO prediction to crop
"""

from ultralytics import YOLO
import os
import cv2 as cv
import matplotlib.pyplot as plt

# Change parameters here
from parameters import *

def cropImage(image, coords): 
    center_x = coords[0] ; center_y = coords[1]
    width = coords[2] ; height = coords[3]

    # Calculate top-left and bottom-right corners
    top_left_x = int(center_x - width // 2)
    top_left_y = int(center_y - height // 2)
    bottom_right_x = int(center_x + width // 2)
    bottom_right_y = int(center_y + height // 2)

    # Perform cropping
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_image

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)  

if __name__ == "__main__": 
    # Declare the image path, image list and output path
    predict_path = os.path.join(os.getcwd(), "test")
    predict_list = os.listdir(predict_path)
    out_path = os.path.join(os.getcwd(), "out_predict") ; createDir(out_path) 

    # Map the full directory to each image
    predict_list = list(map(lambda file: os.path.join(predict_path, file), predict_list))
        
    print("\nStarting prediction...\n")
    print(f"Fetching weights from...{BEST_MODEL_DIR_PREDICT}\n")
    model = YOLO(BEST_MODEL_DIR_PREDICT)

    results = model(predict_list)
    
    # Save the prediction
    for index, result in enumerate(results): 
        boxes = result.boxes
        image = result.orig_img
        coords = boxes.xywh[0]

        cropped_image = cropImage(image, coords)

        cv.imwrite(os.path.join(out_path, f"result_{index}_cropped.jpg"), cropped_image)

        # for box in result.boxes: 
        #     print()
        #     print(box)
            # result.save(filename=os.path.join(out_path, f"result_{index}.jpg"))  # save to disk

    print(f"\nFinished prediction, please check your directory for a file named 'results.jpg'")
