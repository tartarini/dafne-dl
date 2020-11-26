# -*- coding: utf-8 -*-

import SimpleITK as sitk


def biascorrection(nifti_file):
    inputImage = sitk.ReadImage(nifti_file,sitk.sitkFloat32) 
    image = inputImage
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    numberOfIteration = [50] 
    corrector.SetMaximumNumberOfIterations(numberOfIteration * numberFittingLevels)
    output = corrector.Execute(image, maskImage)
    img2=sitk.GetArrayFromImage(output)
    return img2
