import cv2 as cv
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import annotation_tool.matplot_annotate_utils as matplot_annotate_utils

def onClick(event): 
    x, y = int(event.xdata), int(event.ydata)
    if x is not None and y is not None: 
        coordinates.append((x, y))
    
        # Draw the corner dots
        dot, = plt.plot(x, y, "ro")
        objects.append(dot)
        plt.draw()

        if len(objects) > 2: 
            for obj in objects: 
                obj.remove()        # --> remove the dot from the plot
            objects.clear()         # --> clear the objects list
            coordinates.clear()     # --> clear the coordinates list
            print("\nOops, Restart your Corners")
        else: 
            print(f"Placed Corner in ({x}, {y})")

        if len(objects) == 2: 
            # Draw the rectangle and append it to objects (to remove later)
            p3, p4 = matplot_annotate_utils.getOppDiagonals(coordinates[0], coordinates[1])
            points = [coordinates[0], coordinates[1], p3, p4]
            
            # Sorts the points into the respective corner points 
            bottom_left, bottom_right, top_left, top_right = matplot_annotate_utils.getCorners(points) 

            w = bottom_right[0] - bottom_left[0]
            h = top_right[1] - bottom_right[1]

            rect = Rectangle((bottom_left[0], bottom_left[1]),
                             w, h, edgecolor="r", fill=False)  
            print("Annotation (Rectangle) Preview")
        
            ax.add_patch(rect)
            objects.append(rect)

            # Update the state, and normalize the coordinates
            center_x, center_y = matplot_annotate_utils.getCenter(bottom_left, w, h)
            print(center_x, center_y)
            state["coordinates"] = f"{center_x/width} {center_y/height} {w/width} {h/height}"

def onKeyPress(event): 
    labels = ["1", "2", "3", "4", "5", "6"]
    key = event.key

    if key == "enter": 
        print(f"\nAnnotation Saved")
        saveAnnotation()
    elif key in labels: 
        print(f"Class {key} Selected")
        state["class"] = key
    elif key == "escape": 
        print(f"\nPressed {key}. Terminating Program...")
        exit()

def saveAnnotation(): 
    class_id = state["class"]
    coordinates = state["coordinates"]
    format = f"{class_id} {coordinates}"
    for image in images:
        image = os.path.basename(image).split(".JPG")[0]
        image_path = os.path.join(out_path, image)

        # Write the saved annotation to a text file
        f = open(f"{image_path}.txt", "w")
        f.write(format)
        f.close()

    plt.close()
    print(format)
    print(f"Saved to {image_path}.txt\n")

state = {
    "class:": None, 
    "coordinates": []}

if __name__== "__main__": 
    in_dir = "train_annotate"
    out_dir = "train_annotated"

    root = "label_"
    labels = matplot_annotate_utils.getDir(in_dir, root)
    labels_out = matplot_annotate_utils.getDir(out_dir, root)

    for i in range(len(labels)): 
        sids = matplot_annotate_utils.getSidDir(labels[i])

        print(f"\n\n---THIS IS LABEL {os.path.basename(labels[i])}---\n\n")
        for j in range(len(sids)):
            # Create the output directory
            out_path = os.path.join(labels_out[i], os.path.basename(sids[j]))
            matplot_annotate_utils.createDir(out_path)

            images = matplot_annotate_utils.getImages(sids[j])

            image = cv.imread(images[0])
            height, width, _ = image.shape
            
            # fig, ax = plt.subplots(figsize=(12, 10))
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.imshow(image)

            coordinates = []
            objects = []
            cid1 = fig.canvas.mpl_connect("button_press_event", onClick)
            cid2 = fig.canvas.mpl_connect("key_press_event", onKeyPress)
            plt.show()

            fig.canvas.mpl_disconnect(cid1)
            fig.canvas.mpl_disconnect(cid2)
    print(f"\nAnnotation Completed. Please check your working directory for {out_dir}")