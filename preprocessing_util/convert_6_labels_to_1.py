import os
from concurrent.futures import ThreadPoolExecutor

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def replaceLabelWithOne(path): 
    f = open(path, "r")
    read = f.read().split(" ")
    read[0] = "0"               # --> first element is the label
    f.close()
    return " ".join(read)

def writeFile(out_path, string): 
    f = open(out_path, "w")
    f.write(string)
    print(f"Wrote to {out_path}")

if __name__ == "__main__": 
    root = os.path.join("datasets", "2018_YOLO", "labels")
    train, val = os.path.join(root, "train_small"), os.path.join(root, "val_small")

    train_list = os.listdir(train)
    val_list = os.listdir(val)

    # create output directories
    train_out, val_out = os.path.join(root, "train_small_one"), os.path.join(root, "val_small_one")
    createDir(train_out), createDir(val_out)

    with ThreadPoolExecutor(max_workers=6) as executor:
        for txt in train_list[:]: 
            txt_path = os.path.join(train, txt)        
            new_string = replaceLabelWithOne(txt_path)

            # create out_path
            out_path = os.path.join(train_out, txt)
            writeFile(out_path, new_string)

        for txt in val_list[:]: 
            txt_path = os.path.join(val, txt)        
            new_string = replaceLabelWithOne(txt_path)

            # create out_path
            out_path = os.path.join(val_out, txt)
            writeFile(out_path, new_string)








    