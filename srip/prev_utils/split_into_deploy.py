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

def getDeployment(path): 
    return path.split(" ")[0]

def getOrganizedSID(label_dir): 
    image_list = os.listdir(label_dir)

    # Store the image path within a deployment dir
    sid_path = {}         # e.g. ["14434": 14434_SalmonCreek_072418_082118 (1).JPG]

    # Filter by deployment dir
    for file_name in image_list:
        full_file_path = os.path.join(label_dir, file_name)
        sid = getDeployment(file_name)

        if sid in sid_path:
            sid_path[sid].append(full_file_path)
        else:
            sid_path[sid] = [full_file_path]
    
    return sid_path

if __name__  == "__main__": 
    """
    Script used to separate the labeled images into their respective deployments
    folders, for easier YOLO annotations
    """

    in_dir = "2018/fixed"
    out_dir = "updated_2018"

    folder_name = os.path.basename(in_dir)
    sid_path = getOrganizedSID(in_dir)

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

def get_deployment(path): 
    path = os.path.basename(path)
    path = path.split(" ")[0].split("_")
    path = [item for item in path if item != '']
    return "_".join(path)

def get_organized_deployment(deployment_path, filtered_image_paths): 
    for full_file_path in filtered_image_paths: 
        deployment = get_deployment(full_file_path)
        if deployment in deployment_path:
            deployment_path[deployment].append(full_file_path)
        else:
            deployment_path[deployment] = [full_file_path]
    return deployment_path
