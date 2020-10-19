#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:25:52 2020

@author: francesco
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class IncompatibleModelError(Exception):
    pass

class DeepLearningClass(ABC):   
    
    @abstractmethod
    def initModel(self):
        """
        Initializes the model when needed

        Returns
        -------
        None.

        """
        pass
    
    @abstractmethod
    def calcDelta(self, baseModel: DeepLearningClass) -> DeepLearningClass:
        """
        Calculate a delta with another model, Returns a new instance

        Parameters
        ----------
        baseModel : DeepLearningClass
            Base model to calculate the delta from

        Returns
        -------
        DeepLearningClass
            A deep learning class representing the delta of the two models

        """
        pass
    
    def __sub__(self, rhs):
        return self.calcDelta(rhs)
    
    @abstractmethod
    def applyDelta(self, deltaModel: DeepLearningClass) -> DeepLearningClass:
        """
        Applies a delta to this class and returns a new model with the delta applied
        

        Parameters
        ----------
        deltaModel : DeepLearningClass
            Applies a delta to the current model

        Returns
        -------
        DeepLearningClass
            The model that is the current model plus the delta

        """
        pass
    
    def __add__(self, rhs):
        return self.applyDelta(rhs)
    
    @abstractmethod
    def incrementalLearn(self, trainingData, trainingOutputs):
        """
        Perform an incremental learning step on the given training data/outputs

        Parameters
        ----------
        trainingData : TYPE
            Training data.
        trainingOutputs : TYPE
            Training outputs.

        Returns
        -------
        None.

        """
        pass
    
    @abstractmethod
    def apply(self, data: dict):
        """
        Applies the deep learning model to the image

        Parameters
        ----------
        data : Dictionary
            Contains the data and the extra information (for example image and resolution)

        Returns
        -------
        Depends on the operation performed:
            For classifiers: str - Containing the label of the image
            For segmenters: dict[str, mask] - Containing the labels and the corresponding 2D masks

        """
        pass

    def __call__(self, data: dict):
        return self.apply(data)

class ModelProvider(ABC):
    """
        Abstract class that is the base for loading (and, in the future, storing?) models.
        Has to be subclassed to support local and remote loading.
    """
    
    @abstractmethod
    def loadModel(self, modelName: str) -> DeepLearningClass:
        """

        Parameters
        ----------
        modelName : str
            The name of the model to load.

        Returns
        -------
        The weights of the model.

        """
        pass