# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:26:25 2018

@author: AnaFo
"""
import pydicom
import os
#import pandas as pf
import matplotlib.pyplot as plt
import cv2
import numpy as np
#import scipy
from skimage import img_as_float
import imageio



# Converting to Hounsfield units (HU)

#def get_pixels_hu(scans):
#    image= np.stack(scans)
#    #    image = np.stack([s.pixel_array for s in scans])
#    # Convert to int16 (from sometimes int16), 
#    # should be possible as values should always be low enough (<32k)
#    image = image.astype(np.int16)
#
#    # Set outside-of-scan pixels to 1
#    # The intercept is usually -1024, so air is approximately 0
#    image[image == -2000] = 0
#    
#    # Convert to Hounsfield units (HU)
#    intercept = scans[0].RescaleIntercept
#    slope = scans[0].RescaleSlope
#    
#    if slope != 1:
#        image = slope * image.astype(np.float64)
#        image = image.astype(np.int16)
#        
#    image += np.int16(intercept)
#    
#    return np.array(image, dtype=np.int16)

def read_files(patients_folder,folder_path):                  
    for patient in patients_folder[:1]:
        path = folder_path + patient
        all_slices = [pydicom.dcmread(path + '/' + file) for file in os.listdir(path)]
        slices = [s for s in all_slices if s.__contains__("ImagePositionPatient")==True]
        del slices[-1]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) 
    return slices

def normalization(slices):#Converting to float64 and normalization
    normalized=[]
    for each_slice in enumerate(slices):
        
        image = np.array(each_slice[1].pixel_array)
        h = np.max(image)
        l= np.min(image)
        img_norm= (image-l)/(h-l)
        normalized.append(img_norm)
    return normalized

def plotting_12slices(normalized_volume,resized_volume,img_size,a):
    fig = plt.figure()

    
    for plot, each_slice in enumerate(normalized_volume[:12]):
        
        # Proportion
        #d0,d1,d2 = each_slice.meta['sampling']
        #asp=d1/d0
        #normalized_vol.append(img_norm)
        a=each_slice
        #image = img_as_float(each_slice[1])
        resized_image,resized_volume=resize_image(each_slice, img_size,resized_volume)
        y = fig.add_subplot(3,4,plot+1)
        y.imshow(resized_image, cmap='gray') 
    plt.show()
    return a,resized_volume,resized_image
    
def resize_image(image, image_size, resized_volume):
            
    #fig=plt.figure()
    #y = fig.add_subplot(1,2,1)
    #print(image)
    #y.imshow(image)
    resized_image = cv2.resize(image,(image_size,image_size))
    #y = fig.add_subplot(1,2,2)
    #y.imshow(resized_image)
    #plt.show()
    resized_volume=np.dstack((resized_volume,resized_image))
    return resized_image, resized_volume
    

#def main():

folder_path = 'C:/Data/'
patients_folder= os.listdir(folder_path)#+folder_name)                    
img_size = 150
resized_volume=np.zeros((150,150))
a=np.zeros((150,150))

slices = read_files(patients_folder,folder_path)
#plt.imshow(slices[1].pixel_array)
normalized_volume=normalization(slices)
#plt.imshow(normalized_volume[1])
b, vol_resized,img_resized=plotting_12slices(normalized_volume,resized_volume,img_size,a)
#plt.imshow(resized_volume[::12])

#if __name__=="__main__":
 #   main()