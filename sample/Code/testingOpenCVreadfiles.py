# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:26:25 2018

@author: AnaFo
"""
import pydicom
import os
#import pandas as pf
import matplotlib.pyplot as plt
#from matplotlib
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
import numpy as np
#import scipy
from skimage import img_as_float
import imageio
from skimage import measure

resized_volume=[]

folder_path = 'C:/Data/'
#folder_name = "43181879"
patients_folder= os.listdir(folder_path)#+folder_name)
                     
img = imageio.imread('Data/43181879/CT.1.2.840.113619.2.278.3.2831217741.218.1437717384.223.16.dcm')
img_size = 150
normalized_vol=[]

def read_files(path):                  
    for patient in patients_folder[:1]:
        path = folder_path + patient
        all_slices = [pydicom.dcmread(path + '/' + file) for file in os.listdir(path)]
        slices = [s for s in all_slices if s.__contains__("ImagePositionPatient")==True]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) #Ordena os slices
        fig = plt.figure()
        
        resized_volume=np.zeros((150,150))       
        
        for plot, each_slice in enumerate(slices):
    
            #Converting to float64 and normalization
            image = np.array(each_slice.pixel_array)
            image = img_as_float(image)
            
            resized_image= cv2.resize(image,(img_size,img_size))
            resized_volume=np.dstack((resized_volume,resized_image))#resized_volume.append(resized_image)
            #resized_image = cv2.resize(np.array(each_slice.pixel_array),(img_size,img_size))
            
            #y.imshow(resized_image, cmap='gray')
        
        #plot.show()
        return slices, resized_image, resized_volume
    

slices, resized_image, resized_volume=read_files(folder_path)
transposed = resized_volume.transpose(2,1,0)
plt.imshow(resized_volume[:,:,100], cmap=plt.cm.gray)
plt.imshow(transposed[56,:,:], cmap=plt.cm.gray)

#verts, faces = measure.marching_cubes(p, threshold)

#plt.hold(True)

# for i in range(transposed.shape[1]):
#     #image = np.array(slices[0].pixel_array)
# #    h = np.max(transposed[i])
# #    l= np.min(transposed[i])
# #    img_norm= (transposed[i]-l)/(h-l)
#     #normalized_vol.append(img_norm)
#     #image = img_as_float(img_norm)
#     plt.imshow(transposed[i], cmap=plt.cm.gray)
#    # print(i)
#     plt.pause(0.05)
# plt.show()