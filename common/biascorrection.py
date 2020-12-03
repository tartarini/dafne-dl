# -*- coding: utf-8 -*-

import SimpleITK as sitk

def biascorrection(file_or_image):
    if type(file_or_image) == str:
        return biascorrection_file(file_or_image)
    else:
        return biascorrection_image(file_or_image)

def biascorrection_image(image):
    if not type(image) == sitk.SimpleITK.Image:
        image = sitk.GetImageFromArray(image)
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    numberOfIteration = [50]
    corrector.SetMaximumNumberOfIterations(numberOfIteration * numberFittingLevels)
    output = corrector.Execute(image, maskImage)
    img2 = sitk.GetArrayFromImage(output)
    return img2

def biascorrection_file(nifti_file):
    inputImage = sitk.ReadImage(nifti_file,sitk.sitkFloat32) 
    return biascorrection_image(inputImage)
