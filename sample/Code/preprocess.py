
import numpy as np
import os
import scipy.io as spio
import pydicom
import sys

def names_of_slices(folder_path,patient):
    path = slices_path + patient
    all_slices_names = []
    all_slices=[]
    
    for file in os.listdir(path):
        all_slices.append([pydicom.dcmread(path + '/' + file)])
        all_slices_names.append(file)

    slices = []
    slices_names =[]
    i = 0
    for s in all_slices:

        if s[0].__contains__("ImagePositionPatient")==True:
            slices.append([s])
            slices_names.append(all_slices_names[i])
    i = i+1

    print(slices_names)
    return slices,slices_names,all_slices_names

def normalization(slices):#Converting to float64 and normalization
    normalized=[]
    i=0
    for each_slice in reversed(slices):
        image = np.array(each_slice.pixel_array)
        verification_3D = np.array(image.shape).shape[0]
        if verification_3D==2:
            try:
                intercept = each_slice.RescaleIntercept
                slope = each_slice.RescaleSlope
                #print(slope)
            except:
                print('imhere')
                print(slices[0].RescaleIntercept)
                intercept = slices[0].RescaleIntercept
                slope = slices[0].RescaleSlope
            #slope = each_slice[1].RescaleSlope
            new_values = get_pixels_hu(image, intercept,slope)
            normalized.append(new_values)
        else:
            #print(i)
            i=i+1
    return normalized

def get_pixels_hu(scans,intercept,slope):
    #source code: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    image = scans.astype(np.int16)
    image[image == -2000] = 0
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def arrange_slices(folder_path,patient):
    path = folder_path + patient
    all_slices_names = []
    all_slices=[]
    all_slices = [pydicom.dcmread(path + '/' + file) for file in os.listdir(path)]
    #print("all_slices: " + str(len(all_slices)))
    slices = [s for s in all_slices if s.__contains__("ImagePositionPatient")==True]
    #print("slices: " + str(len(slices)))
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    return slices


def creating_mask(indices,slices):
    mask = np.zeros(slices.shape)
    for s in reversed(indices):
        mask[s[0]-1,s[1]-1,s[2]-1]=1
    return mask