import numpy as np
import time
import scipy.ndimage.interpolation as inter
import cv2
import matplotlib.pyplot as plt
from skimage import morphology

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

#__________________________________________________________________________________
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


def grow(img, seed, thresh_value, t,seg):
    """
    code adapted from Matt Hancock
    http://notmatthancock.github.io/2017/10/09/region-growing-wrapping-c.html
    
    img: ndarray, ndim=3
        An image volume.
    
    seed: tuple, len=3
        Region growing starts from this point.

    t: int
        The image neighborhood radius for the inclusion criteria.
        
    thresh: int
        The difference between the point considered and the segmentated 
        volume mean, below which we considered the point part of the segmented 
        volume. If the difference is above the thresh, it's not considered volume
    
    seg: used when we wish to do segmentation with two seeds
    """
    counter = 0

    half_volume = img.shape[1]//2
    #seg = np.zeros(img.shape, dtype=np.bool)
    checked = np.zeros_like(seg)
    max_seeding = 39705+50  #para o 116
    seg[seed] = True
    checked[seed] = True
    needs_check = get_nbhd(seed, checked, img.shape)
    
    
    while len(needs_check) > 0 and counter < max_seeding:
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

        mean_checked = img[seg].mean()
        #difference = mean_checked - img[imin:imax+1, jmin:jmax+1, kmin:kmax+1].mean()
        #difference = img[pt] - img[imin:imax+1, jmin:jmax+1, kmin:kmax+1].mean()
        difference = img[pt] - mean_checked
        

        
        if abs(difference) <= thresh_value:
            # Include the voxel in the segmentation and
            # add its neighbors to be checked.
            
            seg[pt] = True
            counter = counter +1
            needs_check += get_nbhd(pt, checked, img.shape)
        if counter > max_seeding:
            plt.imshow(seg[:,half_volume,:])
            print("não foi possível obter uma segmentação válida")
    plt.imshow(seg[:,half_volume,:])        
    return seg, counter
#__________________________________________________________________________________

def erode_volume(volume,number):
    eroded  = volume[:,:,0]
    for i in range(1,volume.shape[2]):
        img = volume[:,:,i]
        after = morphology.erosion(img,np.ones([number,number]))
        eroded = np.dstack((eroded,after))
    return np.array(eroded)

def dilate_volume(volume, number):
    dilated = volume[:,:,0]
    for i in range(1,volume.shape[2]):
        img = volume[:,:,i]
        after = morphology.dilation(img,np.ones([number,number]))
        dilated = np.dstack((dilated,after))
    return np.array(dilated)

def testing_seeds(seed_1, seed_2, volume, slide):
    volume[tuple(seed_1)] = 5000 
    volume[tuple(seed_2)] = 5000 
    plt.imshow(volume[:,slide,:]) 

    
def resample(volume, scan, new_spacing=[1,1,1]):    
    # Determine current pixel spacing
    slice_thickness = scan[0].SliceThickness
    i = 1
    while len(str(slice_thickness))==0:
        slice_thickness = scan[i].SliceThickness
        print('Im zero')
        i=i+1
    pixel_spacing_1 = scan[0].PixelSpacing[0]
    i = 1
    while len(str(pixel_spacing_1))==0:
        pixel_spacing_1 = scan[i].PixelSpacing[0]
        print('Im zero')
        i=i+1
    i = 1
    pixel_spacing_2 = scan[0].PixelSpacing[1]
    while len(str(pixel_spacing_2))==0:
        pixel_spacing_2 = scan[i].PixelSpacing[1]
        print('Im zero')
        i=i+1
    
    spacing = np.array([slice_thickness,pixel_spacing_1,pixel_spacing_2])
    resize_factor = spacing / new_spacing
    new_real_shape = volume.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / volume.shape
    new_spacing = spacing / real_resize_factor
    
    image = inter.zoom(volume, real_resize_factor, mode='nearest')
    
    return image, new_spacing
