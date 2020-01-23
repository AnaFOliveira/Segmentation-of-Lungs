
#Used code from: http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html

import matplotlib
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import imutils
from skimage import img_as_float
from skimage import exposure, morphology
from read_files import read_files, normalization, resize_volume

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram"""
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def contrast_stretching(img):
    p2, p98 = np.percentile(img, (5, 95))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale

def equalization_hist(img):
    return exposure.equalize_hist(img)

def adaptive_equalization(img):
    return exposure.equalize_adapthist(img, clip_limit=0.03)

def plots_erosion_dilation_contrast(pre_img):

    eroded = morphology.erosion(pre_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([3,3]))
    
    # Display results
    fig = plt.figure(figsize=(25,21))
    axes = np.zeros((2, 6), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 6, 1)
    for i in range(1, 6):
        axes[0, i] = fig.add_subplot(2, 6, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 6):
        axes[1, i] = fig.add_subplot(2, 6, 7+i)


    ax_img, ax_hist, ax_cdf = plot_img_and_hist(pre_img, axes[:, 0])
    ax_img.set_title('Original')
    
    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 6))
    
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(contrast_stretching(pre_img), axes[:, 1])
    ax_img.set_title('Original with contrast')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(eroded, axes[:, 2])
    ax_img.set_title('Eroded')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(contrast_stretching(eroded), axes[:, 3])
    ax_img.set_title('Eroded with contrast')
    
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(dilation, axes[:, 4])
    ax_img.set_title('Eroded and dilated')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(contrast_stretching(dilation), axes[:, 5])
    ax_img.set_title('Eroded, dilated with contrast')


    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 7))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()
    return None
    
def plots_equalization(pre_img):
    fig = plt.figure(figsize=(25,21))
    axes = np.zeros((2, 4), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5+i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(pre_img, axes[:, 0])
    ax_img.set_title('Without erosion, without contrast stretching')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 4))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(pre_img, axes[:, 0])
    ax_img.set_title('Low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(contrast_stretching(pre_img), axes[:, 1])
    ax_img.set_title('Contrast stretching')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(equalization_hist(pre_img), axes[:, 2])
    ax_img.set_title('Histogram equalization')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(adaptive_equalization(pre_img), axes[:, 3])
    ax_img.set_title('Adaptive equalization')
    
    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    fig.tight_layout()
    plt.show()
    return None

def main():
    
    # Load an image
    all_patients_folder_path = 'F:/CTimages/48625475'
    patients_folder= os.listdir(folder_path)                    
    slices,patients = read_files(patients_folder,folder_path)
    patient_slices = slices
    normalized=np.array(normalization(patient_slices))
    resized_volume = resize_volume(normalized,150)

    pre_img = imutils.rotate(resized_volume[:,85,:],180)
    plots_erosion_dilation_contrast(pre_img)
    
if __name__=="__main__":
    main()