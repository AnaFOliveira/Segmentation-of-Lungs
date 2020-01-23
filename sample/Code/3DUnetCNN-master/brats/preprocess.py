"""
This code was adapted from David G Ellis, available at: https://github.com/ellisdg/3DUnetCNN

Tools for converting, normalizing, and fixing the dataset
"""

import glob
import os
import warnings
import shutil
import sys

import SimpleITK as sitk

#from nipype.interfaces.ants import N4BiasFieldCorrection

from train import config
import numpy as np
from preprocess_ipo_utils import read_data, HU_conversion


def append_basename(in_file, append):
    dirname, basename = os.path.split(in_file)
    base, ext = basename.split(".", 1)
    return os.path.join(dirname, base + append + "." + ext)


def get_background_mask(in_folder, out_file, truth_name="GlistrBoost_ManuallyCorrected"):
    """
    This function computes a common background mask for all of the data in a subject folder.
    :param in_folder: a subject folder from the BRATS dataset.
    :param out_file: an image containing a mask that is 1 where the image data for that subject contains the background.
    :param truth_name: how the truth file is labeled int he subject folder
    :return: the path to the out_file
    """
    background_image = None
    for name in config["all_modalities"] + [truth_name]:
        image = sitk.ReadImage(get_image(in_folder, name))
        if background_image:
            if name == truth_name and not (image.GetOrigin() == background_image.GetOrigin()):
                image.SetOrigin(background_image.GetOrigin())
            background_image = sitk.And(image == 0, background_image)
        else:
            background_image = image == 0
    sitk.WriteImage(background_image, out_file)
    return os.path.abspath(out_file)

def convert_image_format(in_file, out_file):
    sitk.WriteImage(sitk.ReadImage(in_file), out_file)
    return out_file


def window_intensities(in_file, out_file, min_percent=1, max_percent=99):
    image = sitk.ReadImage(in_file)
    image_data = sitk.GetArrayFromImage(image)
    out_image = sitk.IntensityWindowing(image, np.percentile(image_data, min_percent), np.percentile(image_data,
                                                                                                     max_percent))
    sitk.WriteImage(out_image, out_file)
    return os.path.abspath(out_file)


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """
#     correct = N4BiasFieldCorrection() 
#     correct.inputs.input_image = in_file
#     correct.inputs.output_image = out_file
#     try:
#         done = correct.run()
#         return done.outputs.output_image
#     except IOError:
#         warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
#                                      "Will try using SimpleITK for bias field correction"
#                                      " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
#                                      " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
    input_image = sitk.ReadImage(in_file, image_type)
    output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
    sitk.WriteImage(output_image, out_file)
    return os.path.abspath(out_file)


def rescale(in_file, out_file, minimum=0, maximum=20000):
    image = sitk.ReadImage(in_file)
    sitk.WriteImage(sitk.RescaleIntensity(image, minimum, maximum), out_file)
    return os.path.abspath(out_file)

def background_to_zero(in_file, background_file, out_file):
    sitk.WriteImage(sitk.Mask(sitk.ReadImage(in_file), sitk.ReadImage(background_file, sitk.sitkUInt8) == 0),
                    out_file)
    return out_file


def check_origin(in_file, in_file2):
    image = sitk.ReadImage(in_file)
    image2 = sitk.ReadImage(in_file2)
    if not image.GetOrigin() == image2.GetOrigin():
        image.SetOrigin(image2.GetOrigin())
        sitk.WriteImage(image, in_file)


def normalize_image(in_file, out_file, patient_id,bias_correction=False):
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        non_corrected=np.copy(in_file)
        np.save(out_file+"/"+patient_id+"_ct",non_corrected)
    return out_file


def get_truth_image(subject_folder, name):
    file_card = os.path.join(subject_folder, name + ".npy")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))


def convert_brats_folder(patient_id, in_folder, out_folder, mask_folder,truth_name='mask_', no_bias_correction_modalities=None):
    
        
    slices, affine,first_slice = read_data(in_folder, True)
    hu_volume=HU_conversion(slices, first_slice)
    hu_array = np.array(hu_volume)
    trans = np.flip(hu_array.transpose(2,1,0),2)
    data = np.flip(trans,0)
    
    #save data
    out_file = os.path.abspath(os.path.join(out_folder))
    perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
    normalize_image(data, out_file, patient_id,bias_correction=perform_bias_correction) #wether to perform or not correction, if false only copies the image    
    del out_file, slices, hu_volume, hu_array, trans, data
    
    #save affine matrix
    out_file = os.path.abspath(os.path.join(out_folder, patient_id+"_affine.npy"))
    np.save(out_file, affine)
    del out_file
    
    # copy the truth file
    try:
        truth_file = get_truth_image(mask_folder, truth_name+patient_id)
    except RuntimeError:
        truth_file = get_truth_image(mask_folder, truth_name.split("_")[0]+patient_id)

    out_file = os.path.abspath(os.path.join(out_folder, patient_id+"_truth.npy"))
    truth = np.load(truth_file,mmap_mode='r')
    np.save(out_file, truth.copy())
    del out_file
    del truth_file
    del truth


def convert_brats_data(data_folder, out_folder, masks_path,overwrite=False, no_bias_correction_modalities=("flair",)):
    """
    Preprocesses the data and writes it to a given output folder. Assumes the original folder structure.
    :param brats_folder: folder containing the original brats data
    :param out_folder: output folder to which the preprocessed data will be written
    :param masks_folder: indices folder containing the masks - needed to only count with the data with ground-truth
    :param overwrite: set to True in order to redo all the preprocessing
    :param no_bias_correction_modalities: performing bias correction could reduce the signal of certain modalities. If
    concerned about a reduction in signal for a specific modality, specify by including the given modality in a list
    or tuple.
    :return:
    """
    
    masks_folder= os.listdir(masks_path)
    for num, subject_folder in enumerate(masks_folder[1:]):
        print(subject_folder)
        #subject = os.path.basename(subject_folder)
        pre_patient_id = subject_folder.split("_", subject_folder.count(subject_folder))[1] 
        patient_id = pre_patient_id.split(".")[0] 
        subject_folder = os.path.basename(patient_id)
        original_folder = os.path.join(data_folder, os.path.basename(os.path.dirname(subject_folder)),
                                                  patient_id)
        preprocessed_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                                  patient_id)
        
        if not os.path.exists(preprocessed_folder) or overwrite: #se não existir, ele cria a pasta
            if not os.path.exists(preprocessed_folder):
                os.makedirs(preprocessed_folder)
                convert_brats_folder(patient_id,original_folder, preprocessed_folder, masks_path,
                                      no_bias_correction_modalities=no_bias_correction_modalities)
                
                
def memory_control():
    pid = os.getpid()
    ps = psutil.Process(pid)
    memoryUse = ps.memory_info()
    print(memoryUse)
    
    
#@profile   
def main():

    data_folder = "G:/CTimages/original/"
    masks_folder ="G:/Masks/pares/"
    out_folder = "G:/CTimages/preprocessed/" 
    
    convert_brats_data(data_folder, out_folder, masks_folder,overwrite=True,no_bias_correction_modalities=False)
    

if __name__ == "__main__":
    main()