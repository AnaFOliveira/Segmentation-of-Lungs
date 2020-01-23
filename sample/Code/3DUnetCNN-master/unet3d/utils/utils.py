import pickle
import os
import collections
import pydicom

import nibabel as nib

from nilearn.image import new_img_like #reorder_img

try:
    from .nilearn_custom_utils.nilearn_utils import crop_img_to
    from .sitk_utils import resample_to_spacing, calculate_origin_offset
except:
    from nilearn_custom_utils.nilearn_utils import crop_img_to
    from sitk_utils import resample_to_spacing, calculate_origin_offset   
import numpy as np
import scipy.ndimage.interpolation as inter


def crop_img_to_numpy_Version(img, affine, slices, copy=True):
    
    """
    Code adapted from nibabel module, available at: https://github.com/nilearn/nilearn/blob/master/nilearn/image/image.py
    
    Crops image to a smaller size
    Crop img to size indicated by slices and adjust affine
    accordingly
    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Img to be cropped. If slices has less entries than img
        has dimensions, the slices will be applied to the first len(slices)
        dimensions
    slices: list of slices
        Defines the range of the crop.
        E.g. [slice(20, 200), slice(40, 150), slice(0, 100)]
        defines a 3D cube
    copy: boolean
        Specifies whether cropped data is to be copied or not.
        Default: True
    Returns
    -------
    cropped_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Cropped version of the input image
    """

    data = img.copy()

    cropped_data = data[tuple(slices)]
    if copy:
        cropped_data = cropped_data.copy()

    linear_part = affine[:3, :3]
    old_origin = affine[:3, 3]
    new_origin_voxel = np.array([s.start for s in slices])
    new_origin = old_origin + linear_part.dot(new_origin_voxel)

    new_affine = np.eye(4)
    new_affine[:3, :3] = linear_part
    new_affine[:3, 3] = new_origin

    return cropped_data

def to_matrix_vector(transform):
    
    """
    Code from nilearn module, available at: https://github.com/nilearn/nilearn/blob/master/nilearn/image/resampling.py
    Split an homogeneous transform into its matrix and vector components.
    The transformation must be represented in homogeneous coordinates.
    It is split into its linear transformation matrix and translation vector
    components.
    This function does not normalize the matrix. This means that for it to be
    the inverse of from_matrix_vector, transform[-1, -1] must equal 1, and
    transform[-1, :-1] must equal 0.
    Parameters
    ----------
    transform: numpy.ndarray
        Homogeneous transform matrix. Example: a (4, 4) transform representing
        linear transformation and translation in 3 dimensions.
    Returns
    -------
    matrix, vector: numpy.ndarray
        The matrix and vector components of the transform matrix.  For
        an (N, N) transform, matrix will be (N-1, N-1) and vector will be
        a 1D array of shape (N-1,).
    See Also
    --------
    from_matrix_vector
    """

    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[0:ndimin, 0:ndimout]
    vector = transform[0:ndimin, ndimout]
    return matrix, vector

# def resample(image, new_shape=[64,64,64]):
#     #Code adapted from Guido Zuidhof, available at: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
#     # Determine current pixel spacing
#     #spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

#     #resize_factor = spacing / new_spacing
#     #new_real_shape = image.shape * resize_factor
#     #new_shape = np.round(new_real_shape)
#     real_resize_factor = new_shape / image.shape
#     #new_spacing = spacing / real_resize_factor
    
#     image = inter.zoom(image, real_resize_factor, mode='nearest')
    
#     return image, new_spacing


def reorder_img_to_Numpy(img,previous_affine, new_shape=(64,64,64),resample=None):
    
    """
    Code adapted from nilearn module, available at: https://github.com/nilearn/nilearn/blob/master/nilearn/image/resampling.py
    Returns an image with the affine diagonal (by permuting axes).
    The orientation of the new image will be RAS (Right, Anterior, Superior).
    If it is impossible to get xyz ordering by permuting the axes, a
    'ValueError' is raised.
        Parameters
        -----------
        img: Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Image to reorder.
        resample: None or string in {'continuous', 'linear', 'nearest'}, optional
            If resample is None (default), no resampling is performed, the
            axes are only permuted.
            Otherwise resampling is performed and 'resample' will
            be passed as the 'interpolation' argument into
            resample_img.
    """

    affine = previous_affine.copy()
    A, b = to_matrix_vector(affine)

    if not np.all((np.abs(A) > 0.001).sum(axis=0) == 1):
        # The affine is not nearly diagonal
        if resample is None:
            raise ValueError('Cannot reorder the axes: '
                             'the image affine contains rotations')
        else:
            # Identify the voxel size using a QR decomposition of the
            # affine
            Q, R = np.linalg.qr(affine[:3, :3])
            target_affine = np.diag(np.abs(np.diag(R))[
                                                np.abs(Q).argmax(axis=1)])
            return resample(img, new_shape)

    axis_numbers = np.argmax(np.abs(A), axis=0)
    data = img
    while not np.all(np.sort(axis_numbers) == axis_numbers):
        first_inversion = np.argmax(np.diff(axis_numbers)<0)
        axis1 = first_inversion + 1
        axis2 = first_inversion
        data = np.swapaxes(data, axis1, axis2)
        order = np.array((0, 1, 2, 3))
        order[axis1] = axis2
        order[axis2] = axis1
        affine = affine.T[order].T
        A, b = to_matrix_vector(affine)
        axis_numbers = np.argmax(np.abs(A), axis=0)

    # Now make sure the affine is positive
    pixdim = np.diag(A).copy()
    if pixdim[0] < 0:
        b[0] = b[0] + pixdim[0]*(data.shape[0] - 1)
        pixdim[0] = -pixdim[0]
        slice1 = slice(None, None, -1)
    else:
        slice1 = slice(None, None, None)
    if pixdim[1] < 0:
        b[1] = b[1] + pixdim[1]*(data.shape[1] - 1)
        pixdim[1] = -pixdim[1]
        slice2 = slice(None, None, -1)
    else:
        slice2 = slice(None, None, None)
    if pixdim[2] < 0:
        b[2] = b[2] + pixdim[2]*(data.shape[2] - 1)
        pixdim[2] = -pixdim[2]
        slice3 = slice(None, None, -1)
    else:
        slice3 = slice(None, None, None)
    data = data[slice1, slice2, slice3]
    #affine = from_matrix_vector(np.diag(pixdim), b)

    return data


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def read_image_files(image_files,image_shape=None, crop=None, label_indices=None):
    """
    :param image_files: 
    :param image_shape: 
    :param crop: 
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return: 
    """
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    affine_list= list()
    for index, image_file in enumerate(image_files): #ct + truth (2 files)
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"
        image, affine = read_image(image_file,image_shape=image_shape, crop=crop, interpolation=interpolation)
        image_list.append(image)
        affine_list.append(affine)
    return image_list, affine_list


def read_image(in_file,image_shape=None, interpolation='linear', crop=None):
    print("Reading: {0}".format(in_file))
    path = 'G:/CTimages/preprocessed/' 
    header_path = 'G:/CTimages/original/'
    
    image = np.load(os.path.abspath(path+in_file))
    affine_prefix = in_file.split("_")[0] 
    previous_affine = np.load(os.path.abspath(path+affine_prefix+'_affine.npy'))
    
    patient_id = in_file.split("\\")[0]
    scan_path = header_path+patient_id+'/'
    first_scan = os.listdir(scan_path)[0]
    scan = pydicom.dcmread(scan_path+first_scan)
                
    
    #image = np.load(os.path.abspath(path+in_file))
    image = fix_shape(image)
    if crop:
        image = crop_img_to_numpy_Version(image, previous_affine, crop, copy=True)
    if image_shape:
        return resize(image,previous_affine, scan, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, previous_affine, scan, new_shape, interpolation="linear"):

    spacing =np.array([scan.SliceThickness,scan.PixelSpacing[0],scan.PixelSpacing[1]])    
    image = reorder_img_to_Numpy(image, previous_affine,resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(spacing, zoom_level)
    new_data = resample_to_spacing(image, spacing, new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(previous_affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, spacing)
    
    return new_data,new_affine