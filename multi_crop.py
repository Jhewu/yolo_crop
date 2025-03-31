"""
This code is for YOLO prediction to crop
"""

from ultralytics import YOLO
import os
import cv2 as cv
import matplotlib.pyplot as plt

# Change parameters here
from parameters import *

# Global Parameters Here
IMG_SIZE = (100, 300)

def cropImageWithCenter(image, coords): 
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

def cropImageWithLeftCorner(image, x, y, sw, sh): 
    return image[y:y+sh, x:x+sw]

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)  

def calcNumRect(lw, lh, sw, sh): 
    num_cols = lw // sw
    num_rows = lh // sh
    return num_cols, num_rows

if __name__ == "__main__": 
    # Declare the image path, image list and output path
    predict_path = os.path.join(os.getcwd(), "test")
    predict_list = os.listdir(predict_path)
    out_path = os.path.join(os.getcwd(), "out_multi") ; createDir(out_path) 

    # Map the full directory to each image
    predict_list = list(map(lambda file: os.path.join(predict_path, file), predict_list))
        
    print("\nStarting prediction...\n")
    print(f"Fetching weights from...{BEST_MODEL_DIR_PREDICT}\n")
    model = YOLO(BEST_MODEL_DIR_PREDICT)

    results = model(predict_list)
    
    # Save the prediction
    for index_p, result in enumerate(results[:]): 
        boxes = result.boxes
        image = result.orig_img
        coords = boxes.xywh[0]

        cropped_image = cropImageWithCenter(image, coords)

        print(cropped_image.shape)

        # Calculate how many smaller rectangle fit within the bigger rectangle
        lh, lw, _ = cropped_image.shape
        sh, sw = IMG_SIZE[0], IMG_SIZE[1]
        num_cols, num_rows = calcNumRect(lw, lh, sw, sh)

        # Get left coordinates for each smaller rectangle
        left_coords = []
        for r in range(num_rows): 
            for c in range(num_cols): 
                y = r * sh
                x = c * sw
                left_coords.append((x, y))

        # Calculate Offset to center all multi-crops
        heigh_pad = int( (lh - (num_rows*sh)) // 2 ) 
        width_pad = int( (lw - (num_cols*sw)) // 2 ) 

        print(heigh_pad, width_pad)

        for index_c, coord in enumerate(left_coords[:]): 
            smaller_crop = cropImageWithLeftCorner(cropped_image, coord[0]+width_pad, coord[1]+heigh_pad, sw, sh)
            print(smaller_crop.shape)
            cv.imwrite(os.path.join(out_path, f"result_{index_p}_cropped_{index_c}.jpg"), smaller_crop)

    print(f"\nFinished prediction, please check your directory for a file named 'results.jpg'")