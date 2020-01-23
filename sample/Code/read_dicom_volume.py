"""
Created on Mon Nov 26 15:26:25 2018

@author: AnaFo
"""
import pydicom
import os
#import pandas as pf
import matplotlib.pyplot as plt
import numpy as np
#import scipy
from skimage import img_as_float
import imageio
import cv2
#import imutils



# Converting to Hounsfield units (HU)
# function doesn't work because all the slices should have the same shape, but they don't
# To resize the image, it's necessary to convert to another type_ Cv_32U ou qq coisa assim, 
#naao consigo encontrar o formato especifico tolerado. Tentar converter imagem para matriz?
#Nao e possivel representar todos as imagens sem o resize 

def get_pixels_hu(scans,intercept,slope):
    #source code: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    #image= np.stack(scans)
    #    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = scans.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    #intercept = scans[0].RescaleIntercept
    #slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def read_files(patients_folder,folder_path):
    patients = []
    for patient in patients_folder[:7]:
        path = folder_path + patient
        all_slices = [pydicom.dcmread(path + '/' + file) for file in os.listdir(path)]
        slices = [s for s in all_slices if s.__contains__("ImagePositionPatient")==True]
        del slices[-1]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) #Sort slices
        patients.append(slices)
    return slices,patients

def normalization(slices):#Converting to float64 and normalization
    normalized=[]
    #global_max = 0
    #global_min = 0
    #for each_slice in enumerate(slices):
     #   image = np.array(each_slice[1].pixel_array)
      #  h = np.max(image)
       # l = np.min(image)
        #if (h>global_max):
         #   global_max= h
        #if (l<global_min):
         #   global_min= l
    for each_slice in enumerate(slices):
        image = np.array(each_slice[1].pixel_array)
        intercept = each_slice[1].RescaleIntercept
        slope = each_slice[1].RescaleSlope
        new_values = get_pixels_hu(image, intercept,slope)
        #img_norm= (image-global_min)/(global_max-global_min)
        normalized.append(new_values)

      # Proportion
        #d0,d1,d2 = each_slice.meta['sampling']
        #asp=d1/d0
    return normalized
    
def resize_volume(normalized_volume,img_size):
    resized_volume=[]
    
    for plot, each_slice in enumerate(normalized_volume):
        resized_image,resized_volume=resize_image(each_slice, img_size,resized_volume)
        
    volume = np.array(resized_volume)
    return volume

def resize_image(image, image_size, resized_volume):
    #Resizing the image
    #fig=plt.figure()
    #y = fig.add_subplot(1,2,1)
    #print(image)
    #y.imshow(image)
    resized_image = cv2.resize(image,(image_size,image_size))
    #y = fig.add_subplot(1,2,2)
    #y.imshow(resized_image)
    #plt.show()
    resized_volume.append(resized_image)
    return resized_image, resized_volume
    
def plotting_12slices(vol_resized):
    fig = plt.figure()  
    for plot, each_slice in enumerate(vol_resized[0:12,:,:]): 
        y = fig.add_subplot(3,4,plot+1)
        y.imshow(each_slice, cmap='gray') 
    plt.show()
    
def transpose(vol_resized):
    #Transpose 
    transposed = vol_resized.transpose(1,2,0) 
    #plt.imshow(vol_resized[4,:,:], cmap=plt.cm.gray)
    #plt.imshow(imutils.rotate(transposed[80,:,:],90), cmap=plt.cm.gray) #it starts to see something from slice 40
    #plt.imshow(resized_volume[::12])    
    
#def main():

folder_path = 'C:/Data/'
folder_name = "43181879"
patients_folder= os.listdir(folder_path)#+folder_name)                    
#img = imageio.imread('/43181879/CT.1.2.840.113619.2.278.3.2831217741.218.1437717384.223.16.dcm')
img_size = 150


slices,patients = read_files(patients_folder,folder_path)
normalized_volume=normalization(slices)
normalized_array = np.array(normalized_volume)
#vol_resized=resize_volume(normalized_volume,img_size)
plotting_12slices(normalized_array)
#transpose(normalized_array)

#if __name__=="__main__":
 #   main()