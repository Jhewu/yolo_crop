"""
Import necessary libraries
"""
import argparse
import glob
import os
import cv2
import time

import exifread
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

import piexif
from PIL import Image, ExifTags
from ultralytics import YOLO
from random import shuffle

from concurrent.futures import ThreadPoolExecutor, as_completed

"""HELPER FUNCTION BELOW"""
def get_sid(path):
    ss,sid = path.split('/'),-1
    if path.split('/')[-1].split('_')[0].isdigit():
        sid = int(path.split('/')[-1].split('_')[0])
    return sid

def get_deploy(path):
    return '_'.join(path.split('/')[-2].split('_')[2:5]).split(' ')[0]

def fix_file_names(path):
    ss   = path.split('/')
    stub = ss[-2]
    if ss[-1].find(stub)>-1: #file name has the folder name in it
        if stub.split('_')[-1].isdigit(): #is the folder name ending with a number
            new_path = stub+'_'+ss[-1].replace(stub,'')
        else:                             #it is ending with an initial
            new_path = stub+'_'+ss[-1].replace(stub + '_', '')
    else:                    #file name doesn't have the folder name in it
        if stub.split('_')[-1].isdigit():
            new_path = ss[-2]+'_'+ss[-1]
        else:
            new_path = '_'.join(ss[-2].split('_')[:-1])+'_'+ss[-1]
    new_path = '/'.join(ss[:-1])+'/'+new_path
    if path!=new_path: os.rename(path,new_path)
    return [path,new_path]

def get_labels(paths,class_idx,offset=-1):
    labels = []
    for path in paths:
        lab = int(path.rsplit('label_')[-1].rsplit('/')[0])
        labels += [class_idx[lab]+offset]
    labels = np.asarray(labels,dtype='uint8')
    return labels

def crop_seg(img,seg):
    if np.abs(seg[0]-seg[2]) > np.abs(seg[1]-seg[3]):    #horizontal
        if np.abs(img.shape[0]-seg[1]) < seg[1]: #bottom orientation
            img1 = img[0:seg[1]+1,:,:]
        else:                                    #top    orientation
            img1 = img[seg[1]:,:,:]
    else:                                                 # vertical
        if np.abs(img.shape[1]-seg[2])<seg[2]:   #right  orientation
            img1 = img[:,0:seg[2]+1,:]
        else:                                    #left   orientation
            img1 = img[:,seg[2]:,:]
    return img1

"""ADDED BY JUN"""
def get_deployment(path): 
    path = os.path.basename(path)
    path = path.split(" ")[0].split("_")
    path = [item for item in path if item != '']
    return "_".join(path)

"""MODIFIED TO IMG INSTEAD OF PATH"""
def read_crop_resize(img,width=600,height=200):
    # img       = cv2.imread(img_path)
    seg_line  = [0,int(img.shape[0]*0.925),img.shape[1],int(img.shape[0]*0.925)]
    clip_img  = crop_seg(img,seg_line)
    new_img   = resize(clip_img,width=width,height=height)
    return new_img

# returns the ref image of the first good ref image: (color, sharp, etc..)
def skip_to_ref(image_list,width,height):
    i,ref = 0,None
    if len(image_list)>0:
        ref = read_crop_resize(image_list[0],height=height,width=width)
        while i<len(image_list) and chroma_dropped(ref): #will find the first one that meets all the checks...
            ref = read_crop_resize(image_list[i],height=height,width=width)
            i += 1
    return ref

def resize(img,width=640,height=480,interp=cv2.INTER_CUBIC):
    h_scale = height/(img.shape[0]*1.0)
    w_scale =  width/(img.shape[1]*1.0)
    if w_scale<h_scale:
        dim = (int(round(img.shape[1]*h_scale)),int(round(img.shape[0]*h_scale)))
        img1 = cv2.resize(img,dim,interpolation=interp)
        d = int(round((img1.shape[1]-width)/2.0))
        img1 = img1[:,d:(img1.shape[1]-d),:]
    else:
        dim = (int(round(img.shape[1]*w_scale)),
               int(round(img.shape[0]*w_scale)))
        img1 = cv2.resize(img,dim,interpolation=interp)
        d = int(round((img1.shape[0]-height)))
        img1 = img1[d:img1.shape[0],:,:]
    img1 = cv2.resize(img1,(width,height),interpolation=interp)
    return img1

def rotate(image,angle):
  image_center = tuple(np.array(image.shape[1::-1])/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image,rot_mat,image.shape[1::-1],flags=cv2.INTER_CUBIC)
  return result

def luma_diff(img1,img2): #green is positive luma and red is negative luma
    lum1 = cv2.cvtColor(img1,cv2.COLOR_BGR2YCrCb)[:,:,0]
    lum2 = cv2.cvtColor(img2,cv2.COLOR_BGR2YCrCb)[:,:,0]
    diff = np.asarray(lum1,dtype=int)-np.asarray(lum2,dtype=int)
    pos  = np.zeros_like(diff)
    pos[diff>0] = diff[diff>0]
    neg  = np.zeros_like(diff)
    neg[diff<0] = np.abs(diff[diff<0])
    bgr = np.zeros_like(img1)
    bgr[:,:,2] = pos[:]
    bgr[:,:,1] = neg[:]
    return bgr

def crop(img,pad=0.2,pixels=None):
    if pixels is None:
        h_pad,w_pad  = int(round(pad*img.shape[0]//2)),int(round(pad*img.shape[1]//2))
        img1 = img[h_pad:(img.shape[0]-h_pad+1),w_pad:(img.shape[1]-w_pad+1),:]
    else:
        img1 = img[pixels:(img.shape[0]-pixels+1),pixels:(img.shape[1]-pixels+1),:]
    return img1

def plot(img,size=(12,9)):
    plt.rcParams["figure.figsize"] = size
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()

def sharpen(image,kernel_size=(5, 5),sigma=1.0,amount=1.0,threshold=0):
    blurred = cv2.GaussianBlur(image,kernel_size,sigma)
    sharpened = float(amount+1)*image - float(amount)*blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image-blurred) < threshold
        np.copyto(sharpened,image,where=low_contrast_mask)
    return sharpened

result_list = []
def collect_results(result):
    result_list.append(result)

def get_label(path):
    label = 0
    if path.find('label_')>-1:
        label = int(path.split('label_')[-1].split('/')[0])
    elif path.endswith('_L0.JPG') or path.endswith('_L1.JPG') or path.endswith('_L2.JPG') or\
            path.endswith('_L3.JPG') or path.endswith('_L4.JPG') or path.endswith('_L5.JPG') or path.endswith('_L6.JPG'):
        label = int(path.split('/')[-1].split('_L')[-1].split('.JPG')[0])
    else:
        if os.path.exists(path):
            try:
                E = piexif.load(path)
                if '0th' in E:
                    t =''
                    if 270 in E['0th']:
                        t = E['0th'][270].decode('ascii')
                    if t.isdigit(): label = int(t)
            except Exception as e:
                print(e,path)
    return label

def read_exif_tags(path,tag_set='all'):
    tags,T = {},{}
    with open(path,'rb') as f: tags = exifread.process_file(f)
    if tag_set=='all': tag_set = set(list(tags.keys()))
    for t in sorted(list(tags.keys())):
        if t in tag_set and type(tags[t]) is not str:
            if type(tags[t]) is not bytes:
                tag_value = tags[t].values
                if type(tag_value) is list: tag_value = ','.join([str(v) for v in tag_value])
                if type(tag_value) is str: tag_value = tag_value.rstrip(' ')
                T[t] = str(tag_value)
    return T

"""ADDED BY JUN"""
def extract_deployment_paths(C, sid, deploy): 
    return [image[-1] for image in C[sid][deploy]]

"""ADDED BY JUN"""
def cropImageWithCenter(image, coords): 
    center_x = coords[0] ; center_y = coords[1]
    width = coords[2] ; height = coords[3]

    # Calculate top-left and bottom-right corners
    top_left_x = int(center_x - width // 2)
    top_left_y = int(center_y - height // 2)
    bottom_right_x = int(center_x + width // 2)
    bottom_right_y = int(center_y + height // 2)

    # Perform cropping
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_image

"""ADDED BY JUN"""
def clean_exif(exif_data):
    cleaned_exif = Image.Exif()
    for tag_id, value in exif_data.items():
        try:
            # Try converting the value to bytes to see if it's valid
            test_exif = Image.Exif()
            test_exif[tag_id] = value
            _ = test_exif.tobytes()
            cleaned_exif[tag_id] = value
        except Exception as e:
            print(f"Skipping EXIF tag {ExifTags.TAGS.get(tag_id, tag_id)} due to error: {e}")
    return cleaned_exif

"""IMPORTANT FUNCTIONS BELOW"""
def chroma_dropped(img,cutoff=3):
    dropped = False
    cvt = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    cr_std = np.std(cvt[:,:,1])
    cb_std = np.std(cvt[:,:,2])
    if cr_std<cutoff and cb_std<cutoff: dropped = True
    return dropped

def too_dark(img,cutoff=40):  #night image was too dark
    dark = False
    luma = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    w,h = luma.shape[0],luma.shape[1]
    z1,z2,z3,z4 = np.mean(luma[:w//2,:h//2]),np.mean(luma[:w//2,h//2:]),np.mean(luma[w//2:,:h//2]),np.mean(luma[w//2:,h//2:])
    zs = [(1 if z1<cutoff else 0),(1 if z2<cutoff else 0),(1 if z3<cutoff else 0),(1 if z4<cutoff else 0)]
    if sum(zs)>3: dark = True
    return dark

def too_light(img,cutoff=180): #overexposure
    light = False
    luma = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    w,h = luma.shape[0],luma.shape[1]
    z1,z2,z3,z4 = np.mean(luma[:w//2,:h//2]),np.mean(luma[:w//2,h//2:]),np.mean(luma[w//2:,:h//2]),np.mean(luma[w//2:,h//2:])
    zs = [(1 if z1>cutoff else 0),(1 if z2>cutoff else 0),(1 if z3>cutoff else 0),(1 if z4>cutoff else 0)]
    if sum(zs)>2: light = True
    return light

def blurred(img,cutoff=75.0,ksize=3):    #too much image blur
    blur = False
    luma = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    w,h = luma.shape[0],luma.shape[1]
    d1 = cv2.Laplacian(luma[:w//2,:h//2],ddepth=cv2.CV_64F,ksize=ksize)
    d2 = cv2.Laplacian(luma[:w//2,h//2:],ddepth=cv2.CV_64F,ksize=ksize)
    d3 = cv2.Laplacian(luma[w//2:,:h//2],ddepth=cv2.CV_64F,ksize=ksize)
    d4 = cv2.Laplacian(luma[w//2:,h//2:],ddepth=cv2.CV_64F,ksize=ksize)
    ds = [(1 if np.std(d1)<cutoff else 0),(1 if np.std(d2)<cutoff else 0),(1 if np.std(d3)<cutoff else 0),(1 if np.std(d4)<cutoff else 0)]
    if sum(ds)>2: blur = True
    return blur

def lens_flare(img,pixel_size=None,verbose=False):
    flare = False
    if pixel_size is None:
        pixel_size = int(round(min(img.shape[0:2])/10))
    area  = int(round(0.25*np.pi*pixel_size**2))
    luma  = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    blur  = cv2.medianBlur(luma,5)
    sharp = sharpen(blur,amount=2.0)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = area
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.8
    detector = cv2.SimpleBlobDetector_create(params)
    for t in [200,195,190,185,180,175,170,165,160]:
        ret, thresh = cv2.threshold(sharp,t,255,cv2.THRESH_BINARY_INV)
        kpts = detector.detect(thresh)
        if len(kpts)>0: break
    if len(kpts)>0: flare = True
    if verbose:
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(img,kpts,np.zeros((1,1)),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plot(blobs)
    return flare

# make more robust with hue rotation ???
def luma_rotated(ref,img,sd=3.0,d_min=2.0,d_max=10.0,steps=25,pad=0.2,verbose=False):
    theta,rot = 0.0,False
    angs = np.arange(d_min,d_max,(d_max-d_min)/steps)
    angs = sorted(sorted(angs)+sorted(-1*angs))[::-1]
    refR,imgA = crop(np.copy(ref),pad),crop(np.copy(img),pad)
    ref_diff = np.std(luma_diff(refR,imgA))
    for ang in angs:
        imgR = crop(rotate(np.copy(img),ang),pad)
        new_diff = np.std(luma_diff(refR,imgR))
        if new_diff+sd<ref_diff: theta = ang
    if (theta<0.0 or theta>0.0): rot = True
    return rot

"""ADDED BY JUN"""
def detect_events_processing(i, imgs, ref, min_size=300):
    bw = drk = lht = blr = flr = rot = False
    
    if len(imgs) > 0:
        x = 1
        if imgs[0].shape[0] > min_size:
            while imgs[0].shape[0] // x > min_size:
                x += 1
        
        if x > 1:
            imgA = resize(imgs[i], imgs[i].shape[1] // x, imgs[i].shape[0] // x)
            refA = resize(ref, ref.shape[1] // x, ref.shape[0] // x)
        else:
            imgA = imgs[i]
            refA = ref
        
        bw = chroma_dropped(imgA)
        drk = too_dark(imgA)
        lht = too_light(imgA)
        blr = blurred(imgA)
        flr = lens_flare(imgA)
        rot = luma_rotated(refA, imgA)
    
    return i, {'dark': drk, 'light': lht, 'bw': bw, 'rotated': rot, 'blurred': blr, 'flared': flr}

"""MODIFIED BY JUN """
def detect_events(imgs, ref, min_size=300):
    events = {'dark': {}, 'light': {}, 'bw': {}, 'rotated': {}, 'blurred': {}, 'flared': {}}
    
    if len(imgs) > 0:
        x = 1
        if imgs[0].shape[0] > min_size:
            while imgs[0].shape[0] // x > min_size:
                x += 1
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(detect_events_processing, i, imgs, ref, min_size): i for i in range(len(imgs))}
        
        for future in as_completed(futures):
            i, event_dict = future.result()
            if event_dict['bw']:
                events['bw'][i] = True
            if event_dict['dark']:
                events['dark'][i] = True
            if event_dict['light']:
                events['light'][i] = True
            if event_dict['rotated']:
                events['rotated'][i] = True
            if event_dict['blurred']:
                events['blurred'][i] = True
            if event_dict['flared']:
                events['flared'][i] = True
    
    return events

"""MODIFIED BY JUN"""
# def detect_events(imgs,ref,min_size=300):
#     x,events = 1,{'dark':{},'light':{},'bw':{},'rotated':{},'blurred':{},'flared':{}}
#     if len(imgs)>0:
#         if imgs[0].shape[0]>min_size:
#             while imgs[0].shape[0]//x>min_size: x+=1
#     for i in imgs:
#         if x>1:
#             imgA = resize(imgs[i],imgs[i].shape[1]//x,imgs[i].shape[0]//x)
#             refA = resize(ref,ref.shape[1]//x,ref.shape[0]//x)
#         else:
#             imgA = imgs[i]
#             refA = ref
#         bw  = chroma_dropped(imgA)
#         drk = too_dark(imgA)
#         lht = too_light(imgA)
#         blr = blurred(imgA)
#         flr = lens_flare(imgA)
#         rot = luma_rotated(refA,imgA)
#         if bw:  events['bw'][i]      = True
#         if drk: events['dark'][i]    = True
#         if lht: events['light'][i]   = True
#         if rot: events['rotated'][i] = True
#         if blr: events['blurred'][i] = True
#         if flr: events['flared'][i]  = True
#     return events

"""ADDED BY JUN TO PARALLELIZE"""
def save_img_exif(i, filtered_image_paths, out_dir, all_image_paths, yolo_sharp_images): 
    # Also preservers the input folder structure for regular SRIP

    image_path = filtered_image_paths[i][0]
    deployment = filtered_image_paths[i][-1]

    # Create the output directory
    dest_dir = os.path.join(out_dir, "passed")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    dest_dir = os.path.join(dest_dir, deployment)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    # Clean up the name from "__" excessive underscores
    image_name_with_ext = os.path.basename(image_path)
    name, ext = os.path.splitext(image_name_with_ext)
    cleaned_name_parts = [item for item in name.split("_") if item != '']
    img_name = f"{dest_dir}/{os.path.basename(f'{"_".join(cleaned_name_parts) + ext.split(' ')[-1]}')}"

    # Extract EXIF data and save it alongside the image
    with Image.open(all_image_paths[i][0]) as img:
        exif_data = img.getexif()

    processed_img_pil = Image.fromarray(yolo_sharp_images[i])
    if exif_data:
        cleaned_exif = clean_exif(exif_data)
        exif_bytes = cleaned_exif.tobytes()
        processed_img_pil.save(img_name, exif=exif_bytes, quality=100)
    else:
        processed_img_pil.save(img_name, quality=100)

    print(f"Wrote image to {img_name}")

"""MODIFIED BY JUN"""
def worker_image_partitions(C, out_dir, width=600, height=200, model_dir=f"model/best.pt", window_size=32, conf=0.7):
    # Load the YOLO model
    model = YOLO(model_dir)

    for sid in C:
        for deploy in sorted(C[sid]):
            # Extracting the deployment paths
            paths = extract_deployment_paths(C, sid, deploy)
            shuffle(paths)
            cropped_images = []

            # Predict detection only on a window (selected number of images) with an confidence threshold
            results = model.predict(paths[:window_size], conf=conf)

            # Check if there's any objects with a prediction, and obtain such coordinates
            coords = None
            for result in results: 
                boxes = result.boxes
                if len(boxes) > 0: 
                    coords = boxes.xywh[0]
                    break
        
            # Process images based on whether a prediction was found
            if coords is not None:
                cropped_images = [cropImageWithCenter(cv2.imread(path), coords) for path in paths]
            else:
                cropped_images = [cv2.imread(path) for path in paths]

            print('processing %s paths for sid=%s, deploy=%s'%(len(paths),sid,deploy))

            # [1] find a starting reference image point::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            ref = skip_to_ref([e for e in cropped_images],width=width,height=height) #[datetime,label,camera,path]

            # [2] read images and detect image events::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            downsized_raw_imgs = {} # --> image used to detect
            yolo_raw_imgs = {} # --> image to save
            
            # Storing the initial image paths [0], and the deployment information [1], but needs to be decluttered
            all_image_paths = {}

            """
            COMMENT: Here's our new approach, we will filter all images using crop
            for faster computation in downsized_raw_imgs, and will modify yolo_raw_imgs, 
            which contains the actual original images to be saved
            """

            """COULD POTENTIALLY PARALLELIZE"""
            for i in range(len(paths)):
                all_image_paths[i] = [C[sid][deploy][i][-1], get_deployment(C[sid][deploy][i][-1])]
                yolo_raw_imgs[i] = cropped_images[i]
                downsized_raw_imgs[i] = read_crop_resize(cropped_images[i],height=height,width=width)

            # [3] find events and partition the images on quality::::::::::::::::::::::::::::::::::::::::
            events = detect_events(downsized_raw_imgs, ref)
            ls = {i:C[sid][deploy][i][1] for i in range(len(C[sid][deploy]))} # get original labels if they exist

            yolo_sharp_images = {}
            filtered_image_paths = {}

            """COULD POTENTIALLY PARALLELIZE"""
            for i in downsized_raw_imgs:
                if i in ls: label = ls[i]
                else:       label = 0
                if i not in events['bw']:
                    if i not in events['blurred']:
                        if i not in events['flared']:
                            if i not in events['dark']:
                                if i not in events['light']:
                                    # passed filters
                                    filtered_image_paths[i] = all_image_paths[i]
                                    yolo_sharp_images[i] = yolo_raw_imgs[i]

            print(f"\nThis is before {len(all_image_paths)}")
            print(f"This is after {len(filtered_image_paths)}\n")

            # [4] saving the "good" images to the passed folder::::::::::::::::::::::::::::::::::::::::::::::::::::

            # Iterate over the keys in filtered_image_paths
            with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your CPU
                futures = []
                for i in filtered_image_paths:
                    future = executor.submit(save_img_exif, i, filtered_image_paths, out_dir, all_image_paths, yolo_sharp_images)
                    futures.append(future)

                # Wait for all tasks to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"An error occurred: {e}")

    return True

def process_image_partitions(T,out_dir,width=600, height=200, model_dir=f"model/best.pt", window_size=32, conf=0.7):
    # Originally, the function used to detect events
    # and perform all of the traditional image enhancements
    # but for mini SRIP, it will just save the passed images

    global result_list

    for cpu in T:  # balanced sid/deployments in ||
        print('dispatching %s images to core=%s'%(T[cpu]['n'],cpu))
        worker_image_partitions(T[cpu]['imgs'], out_dir, width, height, model_dir, window_size, conf)

    return True

# Read all images exif data to order and partition triggered events,returns: order,error,trigs
def process_deploys(S, sid, deploy, time_bin): 
    try:
        data = {'order':[],'exif':[],'trigs':[]}
        start = dt.datetime.strptime(deploy.split('_')[0],'%m%d%y')-dt.timedelta(seconds=1)
        stop  = dt.datetime.strptime(deploy.split('_')[1],'%m%d%y')+dt.timedelta(hours=24)
        #---------------------------------------------------------------------------------
        raw = []
        for i in range(len(S[sid][deploy])):
            path  = S[sid][deploy][i]
            label = get_label(path)
            exif  = read_exif_tags(path)
            if 'Image Make' in exif: ct = exif['Image Make']
            else:                    ct = 'Unknown'
            if 'Image DateTime' in exif: ts = dt.datetime.strptime(exif['Image DateTime'],'%Y:%m:%d %H:%M:%S')
            else:                        ts = dt.datetime.strptime('2020:01:01 00:00:00','%Y:%m:%d %H:%M:%S')
            if ts>=start and ts<=stop:
                raw += [[ts,label,ct,path]]
            else:
                data['exif'] += [[ts,label,ct,path]]
        if len(raw)<1 and len(data['exif'])>1: #try to fix the year?
            print('exif date stamps do not match deployment dates for %s images...'%(len(data['exif'])))
            exif_issues,corrected = data['exif'],[]
            start_e = np.min([e[0] for e in exif_issues])
            stop_e  = np.max([e[0] for e in exif_issues])
            year_diff = int(np.round(np.mean([(start-start_e).total_seconds()/(60*60*24*365),
                                                (stop - stop_e).total_seconds()/(60*60*24*365)])))
            print('attempting to correct exif date stamps using offset=%s years'%year_diff)
            for i in range(len(exif_issues)):
                ts = exif_issues[i][0]
                nt = ts+dt.timedelta(days=365*year_diff)
                if nt>=start and nt<=stop: #does the corrected date fix the issues?
                    corrected += [i]       #save its index so we can still toss some...
                    exif_issues[i][0] = nt #patch the corrected year into the datetime stamp
            issues = sorted(set(range(len(exif_issues))).difference(set(corrected)))
            print('%s corrections were made, %s images remain with unfixable issues...'%(len(corrected),len(issues)))
            data['exif'] = []
            for i in issues:    data['exif'] += [exif_issues[i]]
            for c in corrected: raw += [exif_issues[c]]
        raw = sorted(raw, key=lambda x: x[0])

        # [0] Find the maximal deployment timestamp point (within the hour): camera capture rate and offset
        deploy_days = (stop-start).days
        deploy_hours = deploy_days*24    #maximal number of temporal triggered events
        T = [dt.timedelta(minutes=int(t)) for t in np.arange(0,time_bin+60,time_bin)]
        H = {t:0 for t in T}
        for r in raw:
            _t = dt.timedelta(minutes=r[0].time().minute,seconds=r[0].time().second)
            for t in range(1,len(T),1):
                if _t>=T[t-1] and _t<T[t]:
                    H[T[t-1]] += 1
                    break
        max_ts = [dt.timedelta(0),0]
        for h in H:
            if H[h]>max_ts[1]: max_ts = [h,H[h]]

        # [1] Filter out timestamps in the non-maximal point
        R = []
        for r in raw:
            _t = dt.timedelta(minutes=r[0].time().minute,seconds=r[0].time().second)
            if _t>=max_ts[0] and _t<max_ts[0]+dt.timedelta(minutes=time_bin):
                R += [_t]
        if len(R)>0:
            mean_ts = np.mean(R)
            for r in raw:
                _t = dt.timedelta(minutes=r[0].time().minute,seconds=r[0].time().second)
                if abs(mean_ts-_t).total_seconds()<time_bin*60.0: data['order']   += [r]
                else:                                               data['trigs'] += [r]
        n_error,n_trig,n_tot = len(data['exif']),len(data['trigs']),len(S[sid][deploy])
        if n_error>0:
            print('sid=%s,deploy=%s: %s or %s/%s images had timestamps outside the deployment...'\
                    %(sid,deploy,round(n_error/n_tot,2),n_error,n_tot))
        if n_trig>0:
            print('sid=%s,deploy=%s: %s or %s/%s images were potential triggered events...'\
                    %(sid,deploy,round(n_trig/n_tot,2),n_trig,n_tot))
    except Exception as e:
        print(e)
    return sid, deploy, data

def temporally_order_paths(S, num_workers=6, time_bin=5):
    O = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor: 
        futures = []
        for sid in S: 
            O[sid] = {}
            for deploy in sorted(S[sid]): 
                # Submit the deploy processing task to the executor
                futures.append(executor.submit(process_deploys, S, sid, deploy, time_bin))

        for future in as_completed(futures): 
            try: 
                # Ge the result of the completed future
                sid, deploy, data = future.result()
                if data is not None: 
                    # Update the dictionary O with the processed data
                    O[sid][deploy] = data
            except Exception as e: 
                print(f"Error processing deploy: {e}")
    return O

"""MAIN RUNTIME BELOW"""

if __name__ == "__main__": 
    des="""
    This is a mini-version of Stream River Image Processor 
    (SRIP) from Professor Timothy J. Becker. It's mainly used
    to discard separate usable and unusable data before training
    the YOLO auto-cropper
    """

    # Parse all of the arguments
    parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--in_dir',type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--sids',type=str,help='specific comma seperated sids to process\t[all]')
    parser.add_argument('--width', type=int, default=600, help='image width\t[600]')
    parser.add_argument('--height', type=int, default=200, help='image height\t[200]')
    parser.add_argument('--model_dir', type=str, default="model/best.pt", help='path to the YOLO model directory\t[model/best.pt]')
    parser.add_argument('--window_size', type=int, default=32, help='Only perform YOLO inference on these many images per deployment\t[32]')
    parser.add_argument('--conf', type=float, default=0.7, help='YOLO confidence threshold for cropping\t[0.7]')

    args = parser.parse_args()

    # Set defaults or Raise errors
    if args.in_dir is not None:
        in_dir = args.in_dir
    else: raise IOError
    if args.out_dir is not None:
        out_dir = args.out_dir
    else: raise IOError
    if args.model_dir is not None: 
        model_dir = args.model_dir
    else: raise IOError
    if args.sids is not None:
        sids = [int(sid) for sid in args.sids.split(',')]
    if args.width is not None: 
        width = args.width
    else: width = 600
    if args.height is not None: 
        height = args.height
    else: height = 200
    if args.window_size is not None: 
        window_size = args.window_size
    else: window_size = 32
    if args.conf is not None: 
        conf = args.conf
    else: conf = 0.7

    print(f"Input Directory: {in_dir}")
    print(f"Output Directory: {out_dir}")
    print(f"SIDs to Process: {sids}")
    print(f"Image Width: {width}")
    print(f"Image Height: {height}")
    print(f"Model Directory: {model_dir}")
    print(f"Window Size: {window_size}")
    print(f"Confidence Threshold: {conf}")

    # Fetch and fix file names with the pattern and obtain the sorted raw_paths
    raw_path = in_dir+'/*/*.JPG'
    ffids = [fix_file_names(path) for path in glob.glob(raw_path)]
    raw_paths = sorted(glob.glob(raw_path))

    # Condition to check folder structure of in_dir
    if len(raw_paths)<=0:
        print('raw image data is not ordered by label, proceeding to locate unlabeled JPGs...')
        raw_paths = sorted(glob.glob(in_dir+'/*.JPG'))
        print('located %s unlabeled image paths'%(len(raw_paths)))
    else: print('located %s label ordered image paths'%len(raw_paths))

    # Check for trailing slashes and creating the output directory
    while out_dir[-1]=='/': out_dir = out_dir[:-1]
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    print('output directory has been created at:%s'%out_dir)

    # Create dictionary that stores images by site ID
    S = {}
    for path in raw_paths:
        sid,deploy = get_sid(path), get_deploy(path)
        if sid in S:
            if deploy in S[sid]: S[sid][deploy] += [path]
            else:                S[sid][deploy]  = [path]
        else:                    S[sid]  = {deploy:[path]}
    ss = sorted(S)
    if args.sids is None: sids = ss
    for s in ss:
        if s not in sids: S.pop(s)
    print('%s sids were located among the image paths'%len(S))
    print('%s total deployments were found among the image paths'%(sum([len(S[sid]) for sid in S])))

    # Read exif and time sort with temporal trigger removal routine
    O = temporally_order_paths(S) # This will call get_label and look for exif data...
    for sid in O:
        for deploy in O[sid]:
            if len(O[sid][deploy]['trigs'])>0: #[0]remove triggered events
                trigs_dir,trig = out_dir+'/trigs',1
                if not os.path.exists(trigs_dir): os.mkdir(trigs_dir)
                print('copying %s triggered images to the trigs folder:%s'%(len(O[sid][deploy]['trigs']),trigs_dir))
                for r in O[sid][deploy]['trigs']:
                    img_name = trigs_dir+'/S%s_D%s_I%s.JPG'%(sid,deploy,trig)
                    cv2.imwrite(img_name,cv2.imread(r[-1]))
                    trig += 1
            if len(O[sid][deploy]['exif'])>0: #[0]remove triggered events
                exif_dir,exif = out_dir+'/exif',1
                if not os.path.exists(exif_dir): os.mkdir(exif_dir)
                print('copying %s images with non-valid dates to the exif folder:%s'%\
                      (len(O[sid][deploy]['exif']),exif_dir))
                for r in O[sid][deploy]['exif']:
                    img_name = exif_dir+'/S%s_D%s_I%s.JPG'%(sid,deploy,exif)
                    cv2.imwrite(img_name,cv2.imread(r[-1]))
                    exif += 1

    # This code partitions image data for parallel processing by:
    # 1. Aggregating and sorting images by their count.
    # 2. Distributing the sorted data across available CPUs using round-robin.
    # 3. Organizing partitions by CPU, grouping by 'sid' and 'deploy', and tracking image counts.
    # Finally, it computes the total number of images and outputs the partitioning summary.
    H,P,T = [],{i:[] for i in range(cpus)},{}
    for sid in O:
        for deploy in O[sid]:
            H += [(len(O[sid][deploy]['order']),O[sid][deploy]['order'])]
    H = sorted(H,key=lambda x: x[0])[::-1]
    for i in range(len(H)): P[i%cpus] += [H[i]]
    for cpu in P:
        T[cpu] = {'n':0,'imgs':{}}
        for d in range(len(P[cpu])):
            if P[cpu][d][0]>0: # sids can be duplicates, deploys are unique here...
                sid,deploy = get_sid(P[cpu][d][1][0][-1]), get_deploy(P[cpu][d][1][0][-1])
                if sid not in T[cpu]['imgs']: T[cpu]['imgs'][sid] = {}
                T[cpu]['imgs'][sid][deploy] = P[cpu][d][1]
                T[cpu]['n'] += P[cpu][d][0]
    n_images = sum([T[cpu]['n'] for cpu in T])
    print('partitioned %s total images to %s processors'%(n_images,cpus))

    start = time.time()
    process_image_partitions(T,out_dir, width, height, model_dir, window_size, conf)

    stop  = time.time()
    print('processed %s images in %s sec using %s cpus'%(n_images,round(stop-start,2),cpus))
    print('or %s images per sec'%(n_images/(stop-start)))