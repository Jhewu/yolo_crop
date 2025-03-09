import os
from shutil import copy2

def getLabelPaths(root): 
    label_paths = []
    for i in range(1, 7): 
        path = os.path.join(os.getcwd(), root + str(i))
        label_paths.append(path)
    return label_paths

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def getSID(path): 
    return path.split("_")[0]

def getDeploymentDir(path): 
    return path.split("(")[0]

def getOrganizedSID(label_dir): 
    image_list = os.listdir(label_dir)

    # Store the image path within a deployment dir
    deployment_path = {}         # e.g. [14434_SalmonCreek_072418_082118: 14434_SalmonCreek_072418_082118 (1).JPG]

    # Filter by deployment dir
    for file_name in image_list:
        full_file_path = os.path.join(label_dir, file_name)
        deployment = getDeploymentDir(file_name)

        if deployment in deployment_path:
            deployment_path[deployment].append(full_file_path)
        else:
            deployment_path[deployment] = [full_file_path]
    
    return deployment_path

if __name__  == "__main__": 
    """
    A temporary script used to adapt the 2018 dataset
    after flowlabel_mapper.py has been run, to make it
    compatible with the mini_srip.py or 
    stream_river_image_processor.py
    """

    out_dir = "labeled_2018"
    label_paths = getLabelPaths("label_")

    for label_path in label_paths[:]: 
        folder_name = os.path.basename(label_path)
        deployment_path = getOrganizedSID(label_path)
        deployment_path = list(deployment_path.items())

        for elements in deployment_path: 
            # elements is shaped like this: ('deployment', list of images)
            deployment = elements[0]
            image_list = elements[1]

            # output deployment dir
            deployment_dir = os.path.join(out_dir, folder_name, deployment)
            createDir(deployment_dir)
            
            for image in image_list: 
                image_path = os.path.join(deployment_dir, os.path.basename(image))
                copy2(image, image_path)
