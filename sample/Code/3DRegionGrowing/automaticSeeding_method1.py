#from segmentation_3D_secondversion import resample, grow
from finding_biggest_lung import arrange_slices, normalization, get_pixels_hu
#from read_files import normalization
import scipy.ndimage.interpolation as inter
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as spio
import pydicom
import time

def find_seed(volume, thresh, lung):

    #delta = slices*20//217
    #gamma = rows*20//150
    
    #interval = volume[slices-delta:slices+delta,rows-gamma:rows+gamma,cols-gamma:cols+gamma]
    interval = volume.copy()
#    mask = np.zeros(interval.shape)
  
    
    if lung == 'left':
        cols = volume.shape[2]*2//3
    elif lung == 'right':
        cols = volume.shape[2]//3
        
    inicial_slices= volume.shape[0]*2//3
    slices=inicial_slices
    rows= volume.shape[1]//2
    inicial_seed = tuple([slices,rows,cols])
    mask_HU = np.zeros(volume.shape)
    min_HU = -850
    max_HU = -700
    sup = abs(min_HU-thresh)
    inf = abs(max_HU+thresh)
    mask_HU =np.where((volume >= min_HU) & (volume <= max_HU),volume,False)
    print('min: '+str(inf))
    print('max: '+str(sup))
    ready = False
    second_round= False
    i = 0
    while ready == False:
        value = abs(volume[slices,rows,cols] )
        if value >= inf and value <= sup:
    
            interval[slices,rows,cols]=5000
            seed = tuple([slices,rows,cols])
            ready = True
            #print("I found one")
        else: 
            print(value)
            interval[slices,rows,cols]=5000
            slices=slices+1
            #cols=cols
            if slices>=volume.shape[0]:
                print("no seed was found")
                ready = True
                seed = [0,0,0]
    fig = plt.figure()  
    a = fig.add_subplot(1,1,1)
    a.imshow(interval[:,rows,:]) 
    plt.show()
    if seed == [0,0,0]:
        interval = volume.copy()
        slices = inicial_slices
        while second_round == False:
            print(volume[inicial_slices,rows,cols] >= inf)
            print(volume[inicial_slices,rows,cols] <= sup)
            value = abs(volume[slices,rows,cols] )
            if value >= inf and value <= sup:
                interval[inicial_slices,rows,cols]=5000
                seed = tuple([inicial_slices,rows,cols])
                second_round = True
                print("I found one")
            else: 
                print('got here')
                print(value)
                interval[slices,rows,cols]=5000
                slices=slices-1
                #cols=cols
                if slices<0:
                    print("no seed was found")
                    second_round = True
                    seed = [0,0,0]
    fi = plt.figure()  
    b = fi.add_subplot(1,1,1)
    b.imshow(interval[:,rows,:]) 
    plt.show()
    
    new_seed = tuple(seed)
    fig_1 = plt.figure()  
    fig_1.suptitle("Semente inicial", fontsize=16)
    y = fig_1.add_subplot(1,2,1)
    y.imshow(volume[:,rows,:])
    y.scatter([inicial_seed[2]], [inicial_seed[0]],c='r', s=10)
    x = fig_1.add_subplot(1,2,2)
    x.imshow(mask_HU[:,rows,:])
    x.scatter([inicial_seed[2]], [inicial_seed[0]],c='r',s=10)
    plt.show()
    
    fig_2 = plt.figure()  
    fig_2.suptitle("Semente final", fontsize=16)
    y = fig_2.add_subplot(1,2,1)
    y.imshow(volume[:,rows,:])
    y.scatter([new_seed[2]], [new_seed[0]],c='r', s=10)
    x = fig_2.add_subplot(1,2,2)
    x.imshow(mask_HU[:,rows,:])
    x.scatter([new_seed[2]], [new_seed[0]],c='r',s=10)
    plt.show()

    return seed
    