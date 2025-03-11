import cv2 as cv
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import utils

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
            p3, p4 = utils.getOppDiagonals(coordinates[0], coordinates[1])
            points = [coordinates[0], coordinates[1], p3, p4]
            bottom_left, bottom_right, top_left, top_right = utils.getCorners(points)

            w = bottom_right[0] - bottom_left[0]
            h = top_right[1] - bottom_right[1]

            rect = Rectangle((bottom_left[0], bottom_left[1]),
                             w, h, edgecolor="r", fill=False)  
            print("Annotation (Rectangle) Preview")
        
            ax.add_patch(rect)
            objects.append(rect)

            # Update the state, and normalize the coordinates
            center_x, center_y = utils.getCenter(w, h)
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

def saveAnnotation(): 
    class_id = state["class"]
    coordinates = state["coordinates"]
    format = f"{class_id} {coordinates}"
    plt.close()
    print(format)

state = {
    "class:": None, 
    "coordinates": []}

if __name__== "__main__": 
    root = "label_"
    labels = utils.getDir(root)

    for label in labels: 
        sids = utils.getSidDir(label)
        print(f"\n\n---THIS IS LABEL {os.path.basename(label)}---\n\n")

        for sid in sids: 
            images = utils.getImages(sid)

            image = cv.imread(images[0])
            height, width, _ = image.shape
            
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.imshow(image)

            coordinates = []
            objects = []
            cid1 = fig.canvas.mpl_connect("button_press_event", onClick)
            cid2 = fig.canvas.mpl_connect("key_press_event", onKeyPress)
            plt.show()

            fig.canvas.mpl_disconnect(cid1)
            fig.canvas.mpl_disconnect(cid2)