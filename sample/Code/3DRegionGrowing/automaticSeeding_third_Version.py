
import numpy as np
import matplotlib.pyplot as plt
from finding_biggest_lung import creating_mask
from segmentation_functions import resample
import scipy.io as spio

def find_seed(seeds_path,patient_id,region,normalized_array,slices):
    
    seeds = []
    seed_1 = spio.loadmat(seeds_path+'/'+patient_id+'_seed1', squeeze_me=True)["seed1"]
    seed_2 = spio.loadmat(seeds_path+'/'+patient_id+'_seed2', squeeze_me=True)["seed2"]
     
    if region=='left':
        seeds = [seed_2]
    elif region =='right':
        seeds = [seed_1]
    
    mask = creating_mask(seeds,normalized_array)
    mask_resampled, spacing = resample(mask, slices, [5,5,5])
    j = np.unravel_index(np.argmax(mask_resampled), mask_resampled.shape) #tupla
    #print('j: '+str(j))
    seeds= [j] #lista
    return seeds