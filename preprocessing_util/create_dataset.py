import os
from shutil import copy
from concurrent.futures import ThreadPoolExecutor

def getTrainVal(path): 
    return [os.path.join(path, "train"), os.path.join(path, "val")]

def getImgTxt(path): 
    return [os.path.join(path, "images"), os.path.join(path, "labels")]

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def getLabelPaths(parent, root): 
    label_paths = []
    for i in range(1, 7): 
        path = os.path.join(os.getcwd(), parent, root + str(i))
        label_paths.append(path)
    return label_paths

def getSIDPaths(path): 
    sid_paths = []
    sids = os.listdir(path)
    for sid in sids: 
        sid_paths.append(os.path.join(path, sid))
    return sid_paths

def getSID(path):
    return path.split("_")[0]

def getOrderedSIDImages(sid_paths): 
    # Store the image path within a sid
    image_paths = {}         # e.g. ["14434": 14434_SalmonCreek_072418_082118 (1).JPG, etc...]

    for sid_path in sid_paths:
        image_list = os.listdir(sid_path)
        sid = getSID(os.path.basename(sid_path))

        for file_name in image_list:
            full_file_path = os.path.join(sid_path, file_name)            

            if sid in image_paths:
                image_paths[sid].append(full_file_path)
            else:
                image_paths[sid] = [full_file_path]
        
    return image_paths

def sortDict(dict):
   # Sort the dictionary by its value in ascending order
   sorted_items = sorted(dict.items(), key=lambda item: len(item[1]))
   return sorted_items

def copyToValAndDelete(sorted_images, sorted_txt, val_out):
    # Copy the sid images with the least items to the val directory
    images_to_copy = sorted_images[0][1]
    txts_to_copy = sorted_txt[0][1]

    list(map(copy, images_to_copy, [val_out[0]]*len(images_to_copy)))
    list(map(copy, txts_to_copy, [val_out[1]]*len(txts_to_copy)))

    # Remove the copied SID
    sorted_images.remove(sorted_images[0])
    sorted_txt.remove(sorted_txt[0])

    return sorted_images, sorted_txt

if __name__  == "__main__": 
    """
    Script used to create the working dataset for YOLO training. It takes
    both identical directories containing the images and the txt files, after
    running annotation.py

    It also creates a train/val script. 
    """
    # Input directories
    image_dataset = "train_annotate"
    txt_dataset = "train_annotated"

    # Output directories
    out_dir = os.path.join(os.getcwd(), "train_yolo_dataset")

    # Create the output directories
    splits = getTrainVal(out_dir)
    train_out, val_out = map(getImgTxt, splits)
    list(map(createDir, train_out))
    list(map(createDir, val_out))
    
    # Create the image label paths (cwd/image_data/label_1, etc...)
    image_labels = getLabelPaths(parent=image_dataset, root="label_")

    for label in image_labels: 
        print(f"\nIn label {os.path.basename(label)}")
        image_sid_paths = getSIDPaths(label)

        # Create the annotation paths (cwd/annotation_data/label_1/sids..., etc.)
        txt_sid_paths = getSIDPaths(os.path.join(os.getcwd(), txt_dataset, os.path.basename(label)))

        # Structured like this ["14434": 14434_SalmonCreek_072418_082118 (1).JPG, etc...]
        images_dict = getOrderedSIDImages(image_sid_paths)
        txts_dict = getOrderedSIDImages(txt_sid_paths)

        # Sort the dictionary by its value in ascending order (len of paths)
        sorted_images = sortDict(images_dict)
        sorted_txt = sortDict(txts_dict)

        # Copy to Val and Delete the SID
        sorted_images, sorted_txt = copyToValAndDelete(sorted_images, sorted_txt, val_out)

        for element_i in range( len (sorted_images)): 
            image_sid, image_list = sorted_images[element_i]
            txt_sid, txt_list = sorted_txt[element_i]

            with ThreadPoolExecutor(max_workers=6) as executor: 
                # Copy the images to the respective directories
                for image in image_list: 
                    image_path = os.path.join(train_out[0], os.path.basename(image))
                    executor.submit(copy, image, image_path)

                # Copy the txt to the respective directories
                for txt in txt_list: 
                    txt_path = os.path.join(train_out[1], os.path.basename(txt))
                    executor.submit(copy, txt, txt_path)