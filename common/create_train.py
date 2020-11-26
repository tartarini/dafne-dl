# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import os
import skimage
from skimage.morphology import square
import padorcut
from scipy.ndimage import zoom
import biascorrection


"""
        These functions create the dataset of 2D numpy arrays, containing a stack of image and segmentation.

        Parameters
        ----------
        training_data : Dictionary
            Contains the path to the training data and resolution.
        training_outputs : String
            Contains the path to the training labels.

        Returns
        -------
        Training data, saved as numpy arrays in the folders './train/thigh' and './train/leg'

        It is supposed that 3D nifti for each patient are available in training_data['path'] (images) and training_outputs (segmentations), maybe after 
        the manual correction of Dafne generated segmentations of client patients by the users. It is also supposed that the file names of images and segmentations
        for each patient are the same. 
        This should  be integrated for 3D dicom files also (using utils.dicomUtils.loadDicomFile(dicomPath)). 

        Use img=biascorrection(nifti_dir), instead of img=np.asarray(img_data), if you want to use bias field correction on the training dataset.

"""

def create_train_thigh(training_data, training_outputs):

    MODEL_RESOLUTION = np.array([1.037037, 1.037037])
    MODEL_SIZE = (432, 432)
    resolution = np.array(training_data['resolution'])
    zoomFactor = resolution/MODEL_RESOLUTION
    count_thigh=0
    for patient in os.listdir(training_data['path']):
        count_thigh+=1
        nifti_dir=os.path.join(training_data['path'],patient)
        roi_dir=os.path.join(training_outputs,patient)
        nii=nib.load(nifti_dir) ##
        img_data=nii.get_data() ##
        img=np.asarray(img_data) ##
        #img=biascorrection(nifti_dir) ## bias field correction. Use in substitution of the previous 3 lines. 
        niiroi=nib.load(roi_dir)
        img_dataroi=niiroi.get_data()
        imgroi=np.asarray(img_dataroi)   
        for z in range(imgroi.shape[-1]):
            if int (imgroi[:,:,z].sum())!=0:
                imgroi[:,:,z]=skimage.morphology.area_opening(imgroi[:,:,z],area_threshold=4)
                imgroi[:,:,z]=skimage.morphology.area_closing(imgroi[:,:,z],area_threshold=4)        
                numroi=0
                for class_ in range(13):
                    n_pixels=(imgroi[:,:,z]==float(class_)).sum()
                    if n_pixels>0:
                        numroi+=1
                if (numroi>=7): 
                    img2d=img[:,:,z] ##
                    #img2d=img[z,:,:] ## Use in substitution of the previous line if img=biascorrection(nifti_dir). 
                    img2d = zoom(img2d, zoomFactor) 
                    img2d = padorcut(img2d, MODEL_SIZE)
                    imgroi[:,:,z] = zoom(imgroi[:,:,z], zoomFactor, order=0) 
                    imgroi[:,:,z] = padorcut(imgroi[:,:,z], MODEL_SIZE)
                    conc=np.stack((img2d[::-1,::-1].T,imgroi[::-1,::-1,z].T),axis=-1) ##
                    #conc=np.stack((img2d[::-1,::-1],imgroi[::-1,::-1,z].T),axis=-1) ## Use in substitution of the previous line if img=biascorrection(nifti_dir). 
                    np.save(os.path.join('./train/thigh','train_'+str(count_thigh)),conc)
                    

def create_train_leg(training_data, training_outputs):
    
    MODEL_RESOLUTION = np.array([1.037037, 1.037037])
    MODEL_SIZE = (432, 432)
    resolution = np.array(training_data['resolution'])
    zoomFactor = resolution/MODEL_RESOLUTION
    count_leg=0
    for patient in os.listdir(training_data['path']):
        count_leg+=1
        nifti_dir=os.path.join(training_data['path'],patient)
        roi_dir=os.path.join(training_outputs,patient)
        nii=nib.load(nifti_dir) ##
        img_data=nii.get_data() ##
        img=np.asarray(img_data) ##
        #img=biascorrection(nifti_dir) ## bias field correction. Use in substitution of the previous 3 lines. 
        niiroi=nib.load(roi_dir)
        img_dataroi=niiroi.get_data()
        imgroi=np.asarray(img_dataroi)   
        for z in range(imgroi.shape[-1]):
            if int (imgroi[:,:,z].sum())!=0:
                imgroi[:,:,z]=skimage.morphology.area_opening(imgroi[:,:,z],area_threshold=4)
                imgroi[:,:,z]=skimage.morphology.area_closing(imgroi[:,:,z],area_threshold=4)        
                numroi=0
                for class_ in range(7):
                    n_pixels=(imgroi[:,:,z]==float(class_)).sum()
                    if n_pixels>0:
                        numroi+=1
                if (numroi>=4): 
                    img2d=img[:,:,z] ##
                    #img2d=img[z,:,:] ## Use in substitution of the previous line if img=biascorrection(nifti_dir). 
                    img2d = zoom(img2d, zoomFactor) 
                    img2d = padorcut(img2d, MODEL_SIZE)
                    imgroi[:,:,z] = zoom(imgroi[:,:,z], zoomFactor, order=0) 
                    imgroi[:,:,z] = padorcut(imgroi[:,:,z], MODEL_SIZE)
                    conc=np.stack((img2d[::-1,::-1].T,imgroi[::-1,::-1,z].T),axis=-1) ##
                    #conc=np.stack((img2d[::-1,::-1],imgroi[::-1,::-1,z].T),axis=-1) ## Use in substitution of the previous line if img=biascorrection(nifti_dir). 
                    np.save(os.path.join('./train/leg','train_'+str(count_leg)),conc)
                           
