#import scipy.ndimage.interpolation as inter
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as spio
import preprocess

def counting_mask_size(mask):
    turn_boolean = np.where(mask>0.5, True, False)
    number = np.sum(turn_boolean)
    return number

def main():

    slices_path = "G:/CTimages/"
    indices_path ="G:/Plans/CT/Lungs/ok/Separated Lungs"
    indices_folder= os.listdir(indices_path)

    numero_min_voxels = float('Inf')
    numero_max_voxels = 0
    i=0
    for each_mask in indices_folder[43:44]:
        print(i)
        opened = spio.loadmat(indices_path+'/'+each_mask, squeeze_me=True)["indexes"]
        patient_id = each_mask.split(" ", each_mask.count(each_mask))[0] 
        print(patient_id)
        slices = preprocess.arrange_slices(slices_path,patient_id)

        normalized_volume= preprocess.normalization(slices)
        normalized_array = np.array(normalized_volume)
        print(normalized_array.shape)
        mask = preprocess.creating_mask(opened,normalized_array)
        pix_resampled, spacing = preprocess.resample(mask, slices, [5,5,5])
        slide = pix_resampled.shape[1]*2//3 -1
        plt.imshow(pix_resampled[:,slide,:])
        print(slide-1 )
        this_counter = counting_mask_size(pix_resampled)
        if this_counter>numero_max_voxels:
            numero_max_voxels=this_counter
        if this_counter<numero_min_voxels:
            numero_min_voxels=this_counter
        print('mask size: '+str(this_counter))
        print("max: "+str(numero_max_voxels))
        print("min: " + str(numero_min_voxels))
        i=i+1
        del slices
        del pix_resampled
        #del normalized_array
        del normalized_volume
        fig_2 = plt.figure()  
        fig_2.suptitle("final seed", fontsize=16)
        y = fig_2.add_subplot(1,2,1)
        y.imshow(normalized_array[:,301,:])

        x = fig_2.add_subplot(1,2,2)
        x.imshow(mask[:,211,:])
        
        plt.show()
    print("max global: "+str(numero_max_voxels))
    print("min global: " + str(numero_min_voxels))

if __name__=="__main__":
    main()