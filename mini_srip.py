"""
Import necessary libraries
"""
import argparse
import glob
import os
import time
import util

def srip_filter(): 
    return





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
    parser.add_argument('--label_sub_folders',action='store_true',help='generate subfolders for passed images\t[False]')
    parser.add_argument('--cpus',type=int,help='CPU cores to use for || processing\t[1]')
    args = parser.parse_args()

    # Set defaults or Raise errors
    if args.in_dir is not None:
        in_dir = args.in_dir
    else: raise IOError
    if args.out_dir is not None:
        out_dir = args.out_dir
    else: raise IOError
    if args.sids is not None:
        sids = [int(sid) for sid in args.sids.split(',')]
    if args.cpus is not None:
        cpus = args.cpus
    else: cpus = 1

    """THIS IS ONLY FOR TESTING"""
    # args.sids = [14434] #[19022] #,14434, 14523, 15244]
    # sids = [14434]

    params = {'write_labels':args.label_sub_folders, "out_dir":args.out_dir}
    print('using params:%s'%params)

    # Fetch and fix file names with the pattern and obtain the sorted raw_paths
    raw_path = in_dir+'/*/*.JPG'
    ffids = [util.fix_file_names(path) for path in glob.glob(raw_path)]
    raw_paths = sorted(glob.glob(raw_path))

    # Condition to check folder structure of in_dir
    if len(raw_paths)<=0:
        print('raw image data is not ordered by label, proceeding to locate unlabeled JPGs...')
        raw_paths = sorted(glob.glob(in_dir+'/*.JPG'))
        print('located %s unlabeled image paths'%(len(raw_paths)))
    else: print('located %s label ordered image paths'%len(raw_paths))

    # Check for trailing slashes and creating the output directory
    while out_dir[-1]=='/': out_dir = out_dir[:-1]
    out_dir = os.path.join(out_dir, "out_dir")
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    print('output directory has been created at:%s'%out_dir)

    # Create dictionary that stores images by site ID
    S = {}
    for path in raw_paths:
        sid,deploy = util.get_sid(path), util.get_deploy(path)
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
    O = util.temporally_order_paths(S) # This will call get_label and look for exif data...

    """
    COMMENT: EXAMINE IF ALL OF THIS CODE IS NECESSARY
    """
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
                sid,deploy = util.get_sid(P[cpu][d][1][0][-1]), util.get_deploy(P[cpu][d][1][0][-1])
                if sid not in T[cpu]['imgs']: T[cpu]['imgs'][sid] = {}
                T[cpu]['imgs'][sid][deploy] = P[cpu][d][1]
                T[cpu]['n'] += P[cpu][d][0]
    n_images = sum([T[cpu]['n'] for cpu in T])
    print('partitioned %s total images to %s processors'%(n_images,cpus))

    start = time.time()
    util.process_image_partitions(T,params,cpus=cpus)
    stop  = time.time()
    print('processed %s images in %s sec using %s cpus'%(n_images,round(stop-start,2),cpus))
    print('or %s images per sec'%(n_images/(stop-start)))

    # Creates 
    # if args.label_sub_folders:
    #     I = {}
    #     for img_path in glob.glob(out_dir+'/passed*/*.JPG'):
    #         img_name = img_path.split('/')[-1]
    #         label = util.get_label(img_path)
    #         if label in I: I[label] += [img_path]
    #         else:          I[label]  = [img_path]
    #     for l in I: I[l] = sorted(I[l])

    #     for l in sorted(I):
    #         img_dir = '/'.join(I[l][0].split('/')[:-1])+'/label_%s/'%l
    #         if not os.path.exists(img_dir): os.mkdir(img_dir)
    #         for i in range(len(I[l])):
    #             img_name = I[l][i].split('/')[-1]
    #             os.rename(I[l][i],img_dir+img_name)