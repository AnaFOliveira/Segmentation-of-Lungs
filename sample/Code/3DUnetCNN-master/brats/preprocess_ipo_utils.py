import numpy as np
from os import listdir
import pydicom
import dicom_numpy

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

def HU_conversion(slices, first_slice):#Converting to float64 and normalization
    normalized=[]
    i=0
    for each_slice in reversed(slices):
        image = np.array(each_slice)
        verification_3D = np.array(image.shape).shape[0]
        intercept = first_slice.RescaleIntercept
        slope = first_slice.RescaleSlope
        new_values = get_pixels_hu(image, intercept,slope)
        normalized.append(new_values)
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

def arrange_slices(folder_path):
    
    all_slices = [pydicom.dcmread(folder_path + '/' + file) for file in listdir(folder_path)]
    slices = [s for s in all_slices if s.__contains__("ImagePositionPatient")==True]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    return slices

def read_data(folder_path, rescale = None):
    #code adapted from documentation, available at https://dicom-numpy.readthedocs.io/en/latest/
    
    all_slices = [pydicom.dcmread(folder_path + '/' + file) for file in listdir(folder_path)]
    slices = [s for s in all_slices if (s.__contains__("ImagePositionPatient")==True and s[0x08,0x60].value=='CT')]
    
    try:
        voxel_ndarray, affine = dicom_numpy.combine_slices(slices,rescale)
    except dicom_numpy.DicomImportException as e:
        raise
    return voxel_ndarray, affine,slices[0]