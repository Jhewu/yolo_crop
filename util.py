import os
import exifread
import cv2
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import piexif
from shutil import copy

from concurrent.futures import ThreadPoolExecutor, as_completed

"""
ORGANIZE THE CODE UTIL, AND THEN MOVE SOME OF THE IMPORTANT 
FUNCTIONS TO MINI SRIP
"""

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

def read_crop_resize(img_path,width=600,height=200):
    img       = cv2.imread(img_path)
    seg_line  = [0,int(img.shape[0]*0.925),img.shape[1],int(img.shape[0]*0.925)]
    clip_img  = crop_seg(img,seg_line)
    new_img   = resize(clip_img,width=width,height=height)
    return new_img

#returns the ref image of the first good ref image: (color, sharp, etc..)
def skip_to_ref(paths,width,height):
    i,ref = 0,None
    if len(paths)>0:
        ref = read_crop_resize(paths[0],height=height,width=width)
        while i<len(paths) and chroma_dropped(ref): #will find the first one that meets all the checks...
            ref = read_crop_resize(paths[i],height=height,width=width)
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

#tests:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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

#make more robust with hue rotation ???
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
#tests:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def detect_events(imgs,ref,min_size=300):
    x,events = 1,{'dark':{},'light':{},'bw':{},'rotated':{},'blurred':{},'flared':{}}
    if len(imgs)>0:
        if imgs[0].shape[0]>min_size:
            while imgs[0].shape[0]//x>min_size: x+=1
    for i in imgs:
        if x>1:
            imgA = resize(imgs[i],imgs[i].shape[1]//x,imgs[i].shape[0]//x)
            refA = resize(ref,ref.shape[1]//x,ref.shape[0]//x)
        else:
            imgA = imgs[i]
            refA = ref
        bw  = chroma_dropped(imgA)
        drk = too_dark(imgA)
        lht = too_light(imgA)
        blr = blurred(imgA)
        flr = lens_flare(imgA)
        rot = luma_rotated(refA,imgA)
        if bw:  events['bw'][i]      = True
        if drk: events['dark'][i]    = True
        if lht: events['light'][i]   = True
        if rot: events['rotated'][i] = True
        if blr: events['blurred'][i] = True
        if flr: events['flared'][i]  = True
    return events

result_list = []
def collect_results(result):
    result_list.append(result)

def worker_image_partitions(C, out_dir):
    width                = 600
    height               = 200
    for sid in C:
        for deploy in sorted(C[sid]):
            n = len(C[sid][deploy])
            print('processing %s paths for sid=%s, deploy=%s'%(n,sid,deploy))

            # [1] find a starting reference image point::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            ref = skip_to_ref([e[-1] for e in C[sid][deploy]],width=width,height=height) #[datetime,label,camera,path]

            # [2] read images and detect image events::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            raw_imgs = {}
            original_raw_imgs = {}

            """
            COMMENT: Here's our new approach, we will filter all images using crop
            for faster computation in raw_imgs, and will modify original_raw_imgs, 
            which contains the actual original images to be saved
            """
            for i in range(n):
                img_path = C[sid][deploy][i][-1]
                original_raw_imgs[i] = img_path
                raw_imgs[i] = read_crop_resize(img_path,height=height,width=width)

            # [3] find events and partition the images on quality::::::::::::::::::::::::::::::::::::::::
            events = detect_events(raw_imgs, ref)
            ls = {i:C[sid][deploy][i][1] for i in range(len(C[sid][deploy]))} # get original labels if they exist

            sharp_imgs = {}
            original_sharp_imgs = {}

            for i in raw_imgs:
                if i in ls: label = ls[i]
                else:       label = 0
                if i not in events['bw']:
                    if i not in events['blurred']:
                        if i not in events['flared']:
                            if i not in events['dark']:
                                if i not in events['light']:
                                    # passed filters
                                    sharp_imgs[i] = raw_imgs[i] 
                                    original_sharp_imgs[i] = original_raw_imgs[i]

            # [4] copying the "good" images to the passed folder::::::::::::::::::::::::::::::::::::::::::::::::::::
            for i in original_sharp_imgs:
                if i in ls: label = ls[i] 
                else:       label = 0

                passed_dir = out_dir+'/passed'
                if not os.path.exists(passed_dir): os.mkdir(passed_dir)
                img_name = f"passed/{os.path.basename(original_sharp_imgs[i])}"
                copy(original_sharp_imgs[i], img_name) # --> from the shutil library
                print(f"Copied image to {img_name}")
    return True

def process_image_partitions(T,out_dir,cpus=6):
    # Originally, the function used to detect events
    # and perform all of the traditional image enhancements
    # but for mini SRIP, it will just save the passed images

    global result_list
    p2 = mp.Pool(processes=cpus)

    for cpu in T:  # balanced sid/deployments in ||
        print('dispatching %s images to core=%s'%(T[cpu]['n'],cpu))
        p2.apply_async(worker_image_partitions,
                       args=(T[cpu]['imgs'],out_dir),
                       callback=collect_results)
    p2.close()
    p2.join()
    return True

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