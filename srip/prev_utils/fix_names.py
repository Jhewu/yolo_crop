import os
from shutil import copy2

"""
This python is used to fix the bad names files from
a bug I did not fix
"""
def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def nameFile(file_name): 
    new_file_name = file_name.split(" ")
    return " ".join([new_file_name[0], new_file_name[-1]])
    
if __name__ == "__main__": 
    in_dir = "2018/a"
    out_dir = "2018/fixed"
    createDir(out_dir)

    images = os.listdir(in_dir)

    for old_name in images: 
        new_name = nameFile(old_name)
        old_name_path = os.path.join(in_dir, old_name)
        new_name_path = os.path.join(out_dir, new_name)
        copy2(old_name_path, new_name_path)
