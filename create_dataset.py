import os
from shutil import copy2

def getLabelPaths(parent, root): 
    label_paths = []
    for i in range(1, 7): 
        path = os.path.join(os.getcwd(), parent, root + str(i))
        label_paths.append(path)
    return label_paths

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def getSID(path): 
    return path.split("_")[0]

def getSID(path): 
    return path.split("_")[0]

def getOrganizedSID(label_dir): 
    image_list = os.listdir(label_dir)

    # Store the image path within a deployment dir
    sid_path = {}         # e.g. ["14434": 14434_SalmonCreek_072418_082118 (1).JPG]

    # Filter by deployment dir
    for file_name in image_list:
        full_file_path = os.path.join(label_dir, file_name)
        sid = getSID(file_name)

        if sid in sid_path:
            sid_path[sid].append(full_file_path)
        else:
            sid_path[sid] = [full_file_path]
    
    return sid_path

if __name__  == "__main__": 
    """
    Script used to create the working dataset for YOLO training. It takes
    both identical directories containing the images and the txt files, after
    running annotation.py

    It also creates a train/val script. 
    """
    image_dataset = "data_to_annotate"
    txt_dataset = "annotated_data"

    out_dir = "yolo_dataset"
    
    label_paths = getLabelPaths(parent="out_dir", root="label_")

    for label_path in label_paths[:]: 
        folder_name = os.path.basename(label_path)
        sid_path = getOrganizedSID(label_path)
        sid_path = list(sid_path.items())

        for elements in sid_path: 
            # elements is shaped like this: ('deployment', list of images)
            sid = elements[0]
            image_list = elements[1]

            # output deployment dir
            sid_dir = os.path.join(out_dir, folder_name, sid)
            createDir(sid_dir)
            
            for image in image_list: 
                image_path = os.path.join(sid_dir, os.path.basename(image))
                copy2(image, image_path)
