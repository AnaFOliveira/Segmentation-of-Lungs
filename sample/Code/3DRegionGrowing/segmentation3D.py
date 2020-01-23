from read_files import read_files, normalization, resize_volume
#import imutils
import os
import numpy as np
import time
import scipy.io as spio
import cv2


def resize_mask(mask,img_size):
    resized_volume=[]
    
    for plot, each_slice in enumerate(mask):
        resized_image,resized_volume=resize_mask_slice(each_slice, img_size,resized_volume)
        
    volume = np.array(resized_volume)
    return volume

def resize_mask_slice(image, image_size, resized_volume):
    resized_image = cv2.resize(image,(image_size,image_size))
    resized_volume.append(resized_image)
    return resized_image, resized_volume

def get_nbhd(pt, checked, dims):
    #code from Matt Hancock
    #http://notmatthancock.github.io/2017/10/09/region-growing-wrapping-c.html

    nbhd = []

    if (pt[0] > 0) and not checked[pt[0]-1, pt[1], pt[2]]:
        nbhd.append((pt[0]-1, pt[1], pt[2]))
    if (pt[1] > 0) and not checked[pt[0], pt[1]-1, pt[2]]:
        nbhd.append((pt[0], pt[1]-1, pt[2]))
    if (pt[2] > 0) and not checked[pt[0], pt[1], pt[2]-1]:
        nbhd.append((pt[0], pt[1], pt[2]-1))

    if (pt[0] < dims[0]-1) and not checked[pt[0]+1, pt[1], pt[2]]:
        nbhd.append((pt[0]+1, pt[1], pt[2]))
    if (pt[1] < dims[1]-1) and not checked[pt[0], pt[1]+1, pt[2]]:
        nbhd.append((pt[0], pt[1]+1, pt[2]))
    if (pt[2] < dims[2]-1) and not checked[pt[0], pt[1], pt[2]+1]:
        nbhd.append((pt[0], pt[1], pt[2]+1))

    return nbhd


def grow(img, seed, t):
    """
    code from Matt Hancock
    http://notmatthancock.github.io/2017/10/09/region-growing-wrapping-c.html

    img: ndarray, ndim=3
        An image volume.
    
    seed: tuple, len=3
        Region growing starts from this point.

    t: int
        The image neighborhood radius for the inclusion criteria.
    """
    seg = np.zeros(img.shape, dtype=np.bool)
    checked = np.zeros_like(seg)

    seg[seed] = True
    #seg[seed[0],seed[1],seed[2]] = True
    checked[seed] = True
    needs_check = get_nbhd(seed, checked, img.shape)

    while len(needs_check) > 0:
        pt = needs_check.pop()

        # Its possible that the point was already checked and was
        # put in the needs_check stack multiple times.
        if checked[pt]: continue

        checked[pt] = True

        # Handle borders.
        imin = max(pt[0]-t, 0)
        imax = min(pt[0]+t, img.shape[0]-1)
        jmin = max(pt[1]-t, 0)
        jmax = min(pt[1]+t, img.shape[1]-1)
        kmin = max(pt[2]-t, 0)
        kmax = min(pt[2]+t, img.shape[2]-1)

        if img[pt] >= img[imin:imax+1, jmin:jmax+1, kmin:kmax+1].mean():
            # Include the voxel in the segmentation and
            # add its neighbors to be checked.
            seg[pt] = True
            needs_check += get_nbhd(pt, checked, img.shape)
        
    return seg

folder_path = 'C:/Data/'
patients_folder= os.listdir(folder_path)#+folder_name)                    
img_size = 150 

slices,patients = read_files(patients_folder,folder_path)
normalized_volume=normalization(slices)
#normalized_array = np.array(normalized_volume)
resized_array = resize_volume(normalized_volume,32)
#image_coronal = normalized_array[:,305,:]#imutils.rotate/,180)
indexes = spio.loadmat('lungs_coordinates - 43181879.mat', squeeze_me=True)["indexes"] 

mask = np.zeros(normalized_array.shape)
for s in indexes:
    mask[s[0],s[1],s[2]]=1

resized_mask = resize_mask(mask,32)


## ROI______________________________________
#DEFINING ROI LIMITS
def creating_Roi(indexes):
    min_slice=500
    max_slice=0
    min_row=500
    max_row=0
    min_col=500
    max_col=0
    #i=0
    for j in enumerate(indexes):
        i=j[1]
        if i[0]<min_slice:
            min_slice=i[0]
        if i[0]>max_slice:
            max_slice=i[0]
        if i[1]<min_row:
            min_row=i[1]
        if i[1]>max_row:
            max_row=i[1]
        if i[2]<min_col:
            min_col=i[2]
        if i[2]>max_col:
            max_col=i[2]

    roi = normalized_array[min_slice-5:max_slice+5,min_row-5:max_row+5,min_col-5:max_col+5]
    roi_mask = mask[min_slice-5:max_slice+5,min_row-5:max_row+5,min_col-5:max_col+5]
    
    return roi, roi_mask
##________________________________________________________

#GROWING
seed = [125,15,12]
start = time.time()

seg = grow(roi, tuple(seed), 3)
#seg = grow(resized_mask,tuple(seed),5)
stop = time.time()

print("Elapsed time: %.3f seconds." % (stop - start))
