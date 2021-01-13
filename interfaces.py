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
    def init_model(self):
        """
        Initializes the model when needed

        Returns
        -------
        None.

        """
        pass
    
    @abstractmethod
    def calc_delta(self, baseModel: DeepLearningClass) -> DeepLearningClass:
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
        return self.calc_delta(rhs)
    
    @abstractmethod
    def apply_delta(self, delta_model: DeepLearningClass) -> DeepLearningClass:
        """
        Applies a delta to this class and returns a new model with the delta applied
        

        Parameters
        ----------
        delta_model : DeepLearningClass
            Applies a delta to the current model

        Returns
        -------
        DeepLearningClass
            The model that is the current model plus the delta

        """
        pass
    
    def __add__(self, rhs):
        return self.apply_delta(rhs)
    
    @abstractmethod
    def incremental_learn(self, training_data: dict, training_outputs: string):
        """
        Perform an incremental learning step on the given training data/outputs

        Parameters
        ----------
        training_data : Dictionary
            Contains the path to the training data and resolution.
        training_outputs : String
            Contains the path to the training labels.

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
    def load_model(self, model_name: str) -> DeepLearningClass:
        """

        Parameters
        ----------
        model_name : str
            The name of the model to load.

        Returns
        -------
        The weights of the model.

        """
        pass

    @abstractmethod
    def upload_model(self, model_name: str, model: DeepLearningClass):
        """
        Parameters
        ----------
        model_name : str
            The name of the model to upload.
        model:
            The model to be uploaded
        """
        pass