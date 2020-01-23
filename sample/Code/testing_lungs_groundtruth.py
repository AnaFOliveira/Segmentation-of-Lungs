
import scipy.io as spio
import os
import numpy as np
import matplotlib.pyplot as plt
import time

"""Opens ground truth, checks if it has something (since there were cases with empty structures) and creates a mask.
It returns a list with all the analysed indexes, the masks that have content, the names of the files of indexes added,
the name of the files of indexes deleted and the maximum number of ines that the ground truth can contain"""

folder_path = 'G:/Plans/CT/Lungs/All together'
def read_groundtruth(folder_path):
    contours = []
    indexes = []
    apagados = []
    adicionados = []
    nos = []

    for file in os.listdir(folder_path):
    #indexes = [spio.loadmat(folder_path+'/'+file, squeeze_me=True)["indexes"] for file in os.listdir(folder_path)]
        opened = spio.loadmat(folder_path+'/'+file, squeeze_me=True)["indexes"]
        if opened.shape[0] != 0:
            adicionados.append(file)
            indexes.append(opened)
        if opened.shape[0] == 0:
            apagados.append(file)
    max_rows = 600
    for patient in indexes: #FOR EACH CONTOUR
        
        first_slice=patient[1][0]
        last_slice=0
        first_row=patient[1][1]
        last_row=0
        first_col=patient[1][2]
        last_col=0

        for i in patient: #CHECKS ALL THE INDEXES IN THE CONTOUR
            if i[0]<first_slice:
                first_slice = i[0]
            if i[0]>last_slice:
                last_slice = i[0]
            if i[1]<first_row:
                first_row = i[1]
            if i[1]>last_row:
                last_row = i[1]
            if i[2]<first_col:
                first_col = i[2]
            if i[2]>last_col:
                last_col = i[2]

        #print("first_slice: "+str(first_slice))
        #print("lastslice"+str(last_slice))
        #print("first_row"+str(first_row))
        #print("last_row"+str(last_row))
        #print("first_col"+str(first_col))
        #print("last_col"+str(last_col))
        if last_row<max_rows:
            max_rows = last_row
        slices = last_slice-first_slice
        rows = last_row - first_row
        cols = last_col - first_col

       # print('indices: '+ str([slices,rows,cols]))
        mask = np.zeros([slices,rows,cols])

        for s in patient:
            mask_slice = s[0]- first_slice-1
            mask_row = s[1]-first_row-1
            mask_col = s[2]-first_col-1
            #print(mask_slice)
            #print(mask_row)
            #print(mask_col)
            mask[mask_slice,mask_row,mask_col]=1

        contours.append(mask)
        print('other pacient')
    np.save("All_the_lungs_added", adicionados)
    np.save("All_the_lungs_eraised", apagados)
    np.save("All_the_lungs_indices", indexes)
    np.save("All_the_lungs_masks", contours)
    #np.save("All_the_lungs_nos", contours)
    return indexes,contours, adicionados, max_rows,apagados


patients_folder= os.listdir(folder_path)#+folder_name)   
#indexes = spio.loadmat('lungs_coordinates - 43181879.mat', squeeze_me=True)["indexes"]
indexes,contours, adicionados, max_rows,apagados = read_groundtruth(folder_path)