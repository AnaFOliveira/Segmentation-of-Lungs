from functools import partial

from keras import backend as K
import numpy as np

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


def evaluation_Dice_Jac_TruePR_VS(y_true,y_pred):
    
    TP, TN, FP, FN = compare_volumes_lighter(y_true,y_pred)
    dice = Dice(TP,FP,FN)
    jac = Jac(TP,FP,FN)
    truePr = TruePR(TP,FN)
    vs = VoluM(TP,FP, FN)
    evaluation =dice*0.25 +jac*0.25+truePr*0.25+vs*0.25
    return evaluation

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
