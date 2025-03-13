import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from random import shuffle

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)  

if __name__ == "__main__": 
    IN_DIR = os.path.join("datasets", "2018_YOLO", "images", "train_small_one")
    OUT_DIR = "YOLO_CROPPED"
    MODEL_DIR = "/home/student/Desktop/YOLO/train_yolo12n_2025_03_12_18_09_25/yolo12n_2018_YOLO/weights/best.pt"

    # Load the YOLO Model
    model = YOLO(MODEL_DIR)

    createDir(OUT_DIR)

    image_list = os.listdir(IN_DIR)
    shuffle(image_list)

    index = 0
    for image in image_list[:20]: 
        image_path = os.path.join(IN_DIR, image)
        result = model(image_path)[0]

        boxes = result.boxes  # Boxes object for bounding box outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        result.save(filename=f"{OUT_DIR}/result{index}.jpg")  # save to disk
        index+=1

            

        # print(f"\nThis is prediction {result}")
