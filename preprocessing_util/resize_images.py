import os
import cv2
from concurrent.futures import ThreadPoolExecutor

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def resizeAndSave(image_path, output_path, size):
    img = cv2.imread(image_path)
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized)

if __name__  == "__main__": 
    """
    Since YOLO resizes the images anyway and uses normalized coordinates
    for prediction, we can resize our original dataset, to run it more 
    efficiently
    """

    IMG_SIZE = (640, 640)

    # Designate directories and create output directories
    root = "datasets/2018_YOLO/images"
    train_in, val_in = os.path.join(root, "train_big"), os.path.join(root, "val_big")
    train_out, val_out = os.path.join(root, "train_small"), os.path.join(root, "val_small")
    createDir(train_out) ; createDir(val_out)

    # Create image lists
    train_list, val_list = os.listdir(train_in), os.listdir(val_in)

    with ThreadPoolExecutor(max_workers=6) as executor: 
        # for image in val_list: 
        #     input_image_path = os.path.join(val_in, image)
        #     output_image_path = os.path.join(val_out, image)
        #     executor.submit(resizeAndSave, input_image_path, output_image_path, IMG_SIZE)

        for image in train_list: 
            input_image_path = os.path.join(train_in, image)
            output_image_path = os.path.join(train_out, image)
            executor.submit(resizeAndSave, input_image_path, output_image_path, IMG_SIZE)
