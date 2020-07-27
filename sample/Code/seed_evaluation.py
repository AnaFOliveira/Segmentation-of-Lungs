
from segmentation_functions import resample
from finding_biggest_lung import creating_mask
import numpy as np

def avaliate_seed(seed, indices, slices, normalized_array, region, patient_id):
    
    mask_groundtruth = creating_mask(indices, normalized_array)
    mask_groundtruth_resampled, spacing = resample(mask_groundtruth, slices, [5,5,5])
    mask_groundtruth_resampled_boolean = np.where(mask_groundtruth_resampled>0.5,1,0)
    
    if mask_groundtruth_resampled_boolean[seed]==1:
        print("Well located")
        identification = patient_id+'_'+region
        avaliation ='1'
        result = tuple([identification,avaliation])
    else:
        print("Bad location, "+str(patient_id))
        identification = patient_id+'_'+region
        avaliation ='0'
        result = tuple([identification,avaliation])
    return result
    
