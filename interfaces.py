#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:25:52 2020

@author: francesco
"""
from abc import ABC, abstractmethod

class DeepLearningClass(ABC):   
    
    @abstractmethod
    def loadModel(self):
        """
        Load the model when needed

        Returns
        -------
        None.

        """
        pass
    

class ImageSegmenter(DeepLearningClass):

    @abstractmethod
    def segment(image2D, resolution):
        """
        Perform the segmentation of the image

        Parameters
        ----------
        image2D : Two-dimensional image
            Must be convertible to a numpy array
        resolution : sequence (tuple or array)
            Voxel size of the image for resampling

        Returns
        -------
        dict[str, mask]
            Contains the labels and the corresponding 2D masks

        """
        pass

class ImageClassifier(DeepLearningClass):
    
    @abstractmethod
    def getClassification(image2D, resolution) -> str:
        """
        Perform the classification
        
        Parameters
        ----------
        image2D : Two-dimensional image
            Must be convertible to a numpy array
        resolution : sequence (tuple or array)
            Voxel size of the image for resampling

        Returns
        -------
        str
            Label of the classified object.

        """
        pass
    
    def getSegmenter(label: str) -> ImageSegmenter:
        """
        Get a proper classifier instance for this 

        Parameters
        ----------
        label : str
            A classification label as returned by self.getClassification()

        Returns
        -------
        ImageSegmenter
            An instance of an appropriate segmenter for this classification

        """
        pass