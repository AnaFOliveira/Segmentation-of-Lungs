from finding_biggest_lung import counting_mask_size, creating_mask,arrange_slices,normalization
from segmentation_functions import resample
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as spio
import pydicom
import time
from math import log


def compare_volumes_lighter(volume_truth,volume_seg):

    true_positive_mask = np.logical_and(volume_truth ==1 , volume_seg == 1)
    TP = np.sum(true_positive_mask)
    true_negative_mask = np.logical_and(volume_truth ==0 , volume_seg == 0)
    TN = np.sum(true_negative_mask)

    false_positive_mask = np.logical_and(volume_truth ==0 , volume_seg == 1)
    FP = np.sum(false_positive_mask)
    false_negative_mask = np.logical_and(volume_truth ==1 , volume_seg == 0)
    FN = np.sum(false_negative_mask)
    return TP, TN, FP, FN 

def plot_results_segmentation_image(truth_image_ori, prediction_segment, normal_image):
    #Code adapted from KenobiShan and from kraskevich, 
    #available on https://codereview.stackexchange.com/questions/177898/calculate-true-
    #positive-false-positive-true-negative-and-false-negative-and-co
     
    prediction = prediction_segment.astype(np.uint8)
    truth_image = truth_image_ori.astype(np.uint8)

    output_image = np.empty(shape=(prediction.shape[0], prediction.shape[1], 4), dtype=np.uint8)

    true_positive_mask = np.logical_and(truth_image ==1 , prediction == 1)
    true_negative_mask = np.logical_and(truth_image ==0 , prediction == 0)
    false_positive_mask = np.logical_and(truth_image ==0 , prediction == 1)
    false_negative_mask = np.logical_and(truth_image ==1 , prediction == 0)
    
    background_mask = np.logical_and(truth_image ==0 , prediction == 0)
    
    # B-G-R-A
    red = [255, 0, 0, 255]
    green = [0, 255, 0, 255]
    blue = [0, 0, 255, 255]
    purple =  [128, 0, 255, 255]#blue_whiter [0, 128, 255, 255]
    black = [0, 0, 0, 255]

    output_image[background_mask] = black  
    output_image[true_positive_mask] = blue 
    output_image[true_negative_mask] = black
    output_image[false_positive_mask] = purple
    output_image[false_negative_mask] = red  
    fig_2 = plt.figure(figsize=(22,6))  
    fig_2.suptitle("Resultados Segmentação", fontsize=16)
    y = fig_2.add_subplot(1,2,1)
    y.imshow(normal_image)
    x = fig_2.add_subplot(1,2,2)
    black_patch = mpatches.Patch(color='black', label='True Negative')
    blue_patch = mpatches.Patch(color='blue', label='True Positive')
    red_patch = mpatches.Patch(color='red', label='False Negative')
    purple_patch = mpatches.Patch(color='purple', label='False Positive')
    plt.legend(handles=[black_patch, blue_patch, red_patch, purple_patch])
    x.imshow(output_image)
    return output_image

def plotResultsSegmentationImage_WithContours(truth_image_ori, prediction_segment, normal_image, contours):
    #Code adapted from KenobiShan and from kraskevich, 
    #available on https://codereview.stackexchange.com/questions/177898/calculate-true-positive
    #-false-positive-true-negative-and-false-negative-and-co
     
    prediction = prediction_segment.astype(np.uint8)
    truth_image = truth_image_ori.astype(np.uint8)

    output_image = np.empty(shape=(prediction.shape[0], prediction.shape[1], 4), dtype=np.uint8)

    true_positive_mask = np.logical_and(truth_image ==1 , prediction == 1)
    true_negative_mask = np.logical_and(truth_image ==0 , prediction == 0)
    false_positive_mask = np.logical_and(truth_image ==0 , prediction == 1)
    false_negative_mask = np.logical_and(truth_image ==1 , prediction == 0)
    
    background_mask = np.logical_and(truth_image ==0 , prediction == 0)
    
    # B-G-R-A
    red = [255, 0, 0, 255]
    green = [0, 255, 0, 255]
    blue = [0, 0, 255, 255]
    purple =  [128, 0, 255, 255]
    black = [0, 0, 0, 255]

    output_image[background_mask] = black  
    output_image[true_positive_mask] = blue 
    output_image[true_negative_mask] = black 
    output_image[false_positive_mask] = purple
    output_image[false_negative_mask] = red  
    

    
    fig_2 = plt.figure(figsize=(22,6))  
    fig_2.suptitle("Resultados Segmentação", fontsize=16)
    y = fig_2.add_subplot(1,2,1)
    y.imshow(normal_image, interpolation='nearest')
    for n, contour in enumerate(contours):
        y.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)

    x = fig_2.add_subplot(1,2,2)
    black_patch = mpatches.Patch(color='black', label='True Negative')
    blue_patch = mpatches.Patch(color='blue', label='True Positive')
    red_patch = mpatches.Patch(color='red', label='False Negative')
    purple_patch = mpatches.Patch(color='purple', label='False Positive')
    plt.legend(handles=[black_patch, blue_patch, red_patch, purple_patch])
    x.imshow(output_image)
    return output_image

def compare_volumes(volume_truth,volume_seg):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(volume_truth.shape[0]):
        for j in range(volume_truth.shape[1]):
            for k in range(volume_truth.shape[2]):
                if (volume_truth[i][j][k]==1 or volume_truth[i][j][k]==True) and (volume_seg[i][j][k]==1 or volume_seg[i][j][k]==True):
                    TP = TP+1
                if (volume_truth[i][j][k]==0 or volume_truth[i][j][k]==False) and (volume_seg[i][j][k]==0 or volume_seg[i][j][k]==False):
                    TN = TN+1
                if (volume_truth[i][j][k]==0 or volume_truth[i][j][k]==False) and (volume_seg[i][j][k]==1 or volume_seg[i][j][k]==True):
                    FP = FP+1
                if (volume_truth[i][j][k]==1 or volume_truth[i][j][k]==True)  and (volume_seg[i][j][k]==0 or volume_seg[i][j][k]==False):
                    FN = FN+1
    return TP, TN, FP, FN

def names_masks(indices):
    names = []
    for i in indices:
        split_id = indexes.split(" ", pre_result.count(pre_result))[0]
        print(split_id)
        print(pre_result.split(" ", pre_result.count(pre_result))[1])
        #patient_id = split_id.split("_", split_id.count(split_id))[0]
        
def Dice(tp, fp, fn):
    dice = 2*tp/(2*tp+fp+fn)
    return dice

def Jac(tp, fp, fn):
    jac = tp/(tp+fp+fn)
    return jac

def TruePR(tp,fn):
    tpr = tp/(tp+fn)
    return tpr

def VoluM(tp,fp,fn):
    falses = abs(fn-fp)
    denom = 2*tp+fp+fn
    vs = 1-falses/denom
    return vs

def MutualI(tp,tn,fp,fn):
    #probabilities of these regions
    n=tp+tn+fp+fn
    pSg_1 = (tp+fn)/n
    pSg_2 = (tn+fn)/n
    pSt_1 = (tp+fp)/n
    pSt_2 = (tn+fp)/n
    
    #joint probability
    p_s11_s21 = tp/n
    p_s11_s22 = fn/n
    p_s12_s21 = fp/n
    p_s12_s22 = tn/n
    
    #marginal entropy
    H_sg =-(pSg_1*log(pSg_1) + pSg_2*log(pSg_2))
    H_st =-(pSt_1*log(pSt_1) + pSt_2*log(pSt_2))
    
    #joint entropy
    H_sg_st = -(p_s11_s21*log(p_s11_s21)+p_s11_s22*log(p_s11_s22)+p_s12_s21*log(p_s12_s21)+p_s12_s22*log(p_s12_s22))
    
    MI = H_sg + H_st - H_sg_st
    
    return MI
    
def main():
    slices_path = "G:/CTimages/"
    indices_path ="G:/Plans/CT/Lungs/ok/Separate Lungs/"
    segmentation_results_path = "G:/Plans/CT/Lungs/segmentation_results/"

    segmentation_folder= os.listdir(segmentation_results_path)
    metrics_methods = []
    masks = []
    for metodo in segmentation_folder[3:4]: #pasta dos métodos 
        print(metodo)
        results_path = segmentation_results_path+metodo+'/'
        resultados = os.listdir(results_path)
        a = 8
        metrics = []
        TP_list = []
        TN_list = []
        FP_list = []
        FN_list = []
        for num,pre_result in enumerate(resultados[a:9]): #folder of results for each method

            print('status: '+str(num)+' in '+str(len(resultados)))
            print('nome pre-result: ' +str(pre_result))
            split_id = pre_result.split('_', pre_result.count(pre_result))[1] 
            result_id = split_id.split('_', split_id.count(split_id))[0]
            print('paciente_result: '+str(result_id))
            indexes = os.listdir(indices_path)
            indice_id = indexes[a].split(" ")[0]
            region = indexes[a].split("_")[1]
            print('region_indice: '+str(region))

            print('indice_mask: '+str(indice_id))

            #Reading data--------------------------------------------------------------------------------------------------
            indexes_mask = spio.loadmat(indices_path+ indexes[a], squeeze_me=True)["indexes"] 
            slices = arrange_slices(slices_path,result_id)
            normalized_volume=normalization(slices)
            normalized_array = np.array(normalized_volume)
            volume_resampled,spacing = resample(normalized_array, slices, [5,5,5])

            # Ground Truth
            ground_truth = creating_mask(indexes_mask,normalized_array)
            ground_truth_resampled,spacing = resample(ground_truth, slices, [5,5,5])
            ground_truth_boolean = np.where(ground_truth_resampled>0.5, 1, 0)

            # Segmentation results
            result_segment= np.load(results_path +pre_result)
            result_segment_boolean= np.where(result_segment==True, 1, 0)

            print('shape segmentação: '+str(result_segment_boolean.shape))
            print('shape ground truth: '+str(ground_truth_boolean.shape))

            print('\n'+'-----------------------------------------------------------------------------------------')

            #Metrics--------------------------------------------------------------------------------------------------------
            #start_1=time.time()
            #TP_1, TN_1, FP_1, FN_1 = compare_volumes(mask_resampled_boolean,result_boolean)
            #stop_1=time.time()
            #print(TP_1, TN_1, FP_1, FN_1)
            #print("Elapsed time: %.3f seconds." % (stop_1 - start_1))
            start=time.time()
            TP, TN, FP, FN = compare_volumes_lighter(ground_truth_boolean,result_segment_boolean)
            stop=time.time()
            TP_list.append(TP)
            FP_list.append(FP)
            TN_list.append(TN)
            FN_list.append(FN)
            name = segmentation_results_path+"Metrics_Tp_Fp, Tn, Fn"
            #np.save(name, [TP_list,FP_list,TN_list,FN_list])
            #def compare_volumes_lighter(volume_truth,volume_seg):

            print('TP, TN, FP, FN: '+str([TP, TN, FP, FN]))
            print("Elapsed time: %.3f seconds." % (stop - start))
            half_rows = result_segment_boolean.shape[1]//2

            #def plot_results_segmentation_image(truth_image, prediction, normal_image):
            img = plot_results_segmentation_image(ground_truth_boolean[:,half_rows,:],result_segment_boolean[:,half_rows,:],volume_resampled[:,half_rows,:])
            fig = plt.figure(figsize=(22,6))  
            fig.suptitle("Resultados Segmentação", fontsize=16)
            y = fig.add_subplot(1,2,1)
            y.imshow(result_segment_boolean[:,half_rows,:])
            y.set_title("Segmentation results")
            x = fig.add_subplot(1,2,2)
            x.imshow(ground_truth_boolean[:,half_rows,:])
            x.set_title("Ground truth")
            plt.show()

            identification = indice_id+'_'+region
            dice = Dice(TP,FP,FN)
            jac = Jac(TP,FP,FN)
            truePr = TruePR(TP,FN)
            vs = VoluM(TP,FP, FN)
            mi = MutualI(TP,TN,FP,FN)
            avaliation =tuple([dice,jac,truePr,vs, mi])
            result = tuple([identification,avaliation])
            metrics.append(result)

            print('Metrics:')
            print('Dice: '+str(dice))
            print('Jacquard Index: '+str(jac))
            print('True Positive Rate: '+str(truePr))
            print('Volumetric Similarity: '+str(vs))
            print('Mutual Information: '+str(mi))
            print('\n'+'-----------------------------------------------------------------------------------------')
            print('----------------------------------------------------------------------------------------------')

            a=a+1
        metrics_methods.append(metrics)
        