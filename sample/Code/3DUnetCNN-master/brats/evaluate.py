import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from unet3d.metrics import evaluation_Dice_Jac_TruePR_VS, dice_coefficient
import numpy as np


def get_lung_mask(data):
    return data == 1 

def get_background_mask(data):
    return data < 1

# def get_whole_tumor_mask(data):
#     return data > 0


# def get_tumor_core_mask(data):
#     return np.logical_or(data == 1, data == 4)


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


def main():
    header = ("Lungs", "Background")
    masking_functions = (get_lung_mask, get_background_mask)
    metrics = (evaluation_Dice_Jac_TruePR_VS, dice_coefficient)
    metric_name = ("General Avaliation (Dice, Jac, TruePr, Vs)","Dice coefficient")
    rows = list()
    subject_ids = list()
    for num, metric in enumerate(metrics): 
        for case_folder in glob.glob("prediction/*"):
            if not os.path.isdir(case_folder): 
                continue
            subject_ids.append(os.path.basename(case_folder))
            truth_file = os.path.join(case_folder, "{0}_truth.nii.gz")
            truth_image = nib.load(truth_file)
            truth = truth_image.get_data()
            prediction_file = os.path.join(case_folder, "{0}_prediction.nii.gz") 
            prediction_image = nib.load(prediction_file)
            prediction = prediction_image.get_data()

            rows.append([metric(func(truth), func(prediction))for func in masking_functions])

        df = pd.DataFrame(rows, columns=header, index=subject_ids)
        df.to_csv("./prediction/ipo_scores"+metric_name[num]+".csv")
    
        scores = dict()

        for index, score in enumerate(df.columns):
            values = df.values.T[index]
            scores[score] = values[np.isnan(values) == False]

        plt.boxplot(list(scores.values()), labels=list(scores.keys()))
        plt.ylabel(metric_name[num])
        plt.savefig("validation_scores_boxplot"+metric_name[num]+".png")
        plt.close()
        
        del rows
        del subject_ids
        rows = list()
        subject_ids = list()

    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')

        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph.png')


if __name__ == "__main__":
    main()
