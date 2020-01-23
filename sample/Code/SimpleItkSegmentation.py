# %load segmentation.py


# -*- coding: ascii -*-

#import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import os
import imutils
#from skimage import img_as_float, color
from skimage import exposure, morphology
from read_files import read_files, normalization, resize_volume
from read_contours import resize_mask
import cv2


folder_path = 'C:/Data/'
patients_folder= os.listdir(folder_path)                  
slices, patients = read_files(patients_folder,folder_path)
normalized=np.array(normalization(slices))
resized_volume = resize_volume(normalized,150)

pre_img =np.array(imutils.rotate(normalized[:,273,:],180),dtype = np.uint8)

eroded = morphology.erosion(pre_img,np.ones([3,3]))
dilation = morphology.dilation(eroded,np.ones([3,3]))

plt.imshow(eroded)
indexes = spio.loadmat('Right_Lung_coordenates - 43181879.mat', squeeze_me=True)["indexes"] 
seed = [indexes[45]]

# Mask
mask = np.zeros(normalized.shape,dtype = np.uint8)
for s in indexes:
    mask[s[0],s[1],s[2]]=255
mask_volume=np.array(resize_mask(mask,150))
mask_slice=mask_volume[:,85,:]

#imgSmooth = SimpleITK.CurvatureFlow(image1=pre_img,
#                                    timeStep=0.125,
#                                    numberOfIterations=5)