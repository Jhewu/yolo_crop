import os

"""Functions below, are for obtaining data directories"""
def getDir(in_dir, root): 
    dirs = []
    for i in range(1,7):
        dirs.append(os.path.join(os.getcwd(), in_dir, f"{root}{i}"))
    return dirs

def getSidDir(path): 
    dirs = []
    sids = os.listdir(path)
    for i in range(len(sids)): 
        dirs.append(os.path.join(path, sids[i]))
    return dirs

def getImages(path): 
    image_list = []
    images = os.listdir(path)
    for i in range(len(images)): 
        image_list.append(os.path.join(path, images[i]))
    return image_list
"""Functions above, are for obtaining data directories"""

"""Functions below, are for calculating rectangles"""
def getOppDiagonals(p1, p2): 
    p3 = p2[0], p1[1]
    p4 = p1[0], p2[1]
    return p3, p4

def getCorners(points): 
    sorted_p = sorted(points, key=lambda p: p[1])
    bottom_p = sorted_p[:2] ; top_p = sorted_p[2:]

    bottom_left, bottom_right = sorted(bottom_p, key=lambda p: p[0])
    top_left, top_right = sorted(top_p, key=lambda p: p[0])
    return bottom_left, bottom_right, top_left, top_right

def getCenter(width, height): 
    return width // 2, height // 2

"""Functions above, are for calculating rectangles"""

"""Miscellaneous functions"""

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

if __name__  == "__main__": 
    print("You're not supposed to be running this!")