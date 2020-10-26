#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a deep learning module that can be serialized and deserialized, and dynamically changed.
Functions for the operation of the class are provided as references to top-level functions.
Such top level functions should define all the imports within themselves (i.e. don't put the imports at the top of the file).

@author: francesco
"""
from __future__ import annotations
from .interfaces import IncompatibleModelError, DeepLearningClass
import dill
from io import BytesIO

def default_keras_weights_to_model_function(modelObj: DynamicDLModel, weights):
    modelObj.model.set_weights(weights)
    
def default_keras_model_to_weights_function(modelObj: DynamicDLModel):
    return modelObj.model.get_weights()

def default_keras_delta_function(lhs: DynamicDLModel, rhs: DynamicDLModel):
    if lhs.model_id != rhs.model_id: raise IncompatibleModelError
    lhs_weights = lhs.get_weights()
    rhs_weights = rhs.get_weights()
    newWeights = []
    for depth in range(len(lhs_weights)):
        newWeights.append(lhs_weights[depth] - rhs_weights[depth])
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(newWeights)
    return outputObj

def default_keras_apply_delta_function(lhs: DynamicDLModel, rhs: DynamicDLModel):
    if lhs.model_id != rhs.model_id: raise IncompatibleModelError
    lhs_weights = lhs.get_weights()
    rhs_weights = rhs.get_weights()
    newWeights = []
    for depth in range(len(lhs_weights)):
        newWeights.append(lhs_weights[depth] + rhs_weights[depth])
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(newWeights)
    return outputObj

class DynamicDLModel(DeepLearningClass):
    """
    Class to represent a deep learning model that can be serialized/deserialized
    """
    def __init__(self, model_id, # a unique ID to avoid mixing different models
                 init_model_function, # inits the model. Accepts no parameters and returns the model
                 apply_model_function, # function that applies the model. Has the object, and image, and a sequence containing resolutions as parameters
                 weights_to_model_function = default_keras_weights_to_model_function, # put model weights inside the model.
                 model_to_weights_function = default_keras_model_to_weights_function, # get the weights from the model in a pickable format
                 calc_delta_function = default_keras_delta_function, # calculate the weight delta
                 apply_delta_function = default_keras_apply_delta_function, # apply a weight delta
                 incremental_learn_function = None, # function to perform an incremental learning step
                 weights = None): # initial weights
        self.model = None
        self.model_id = model_id
        self.init_model_function = init_model_function
        self.weights_to_model_function = weights_to_model_function
        self.model_to_weights_function = model_to_weights_function
        self.apply_model_function = apply_model_function
        self.calc_delta_function = calc_delta_function
        self.apply_delta_function = apply_delta_function
        self.incremental_learn_function = incremental_learn_function
        self.init_model() # initializes the model
        if weights: self.set_weights(weights)
        
    def init_model(self):
        """
        Initializes the internal model

        Returns
        -------
        None.

        """
        self.model = self.init_model_function()
        
    def set_weights(self, weights):
        """
        Loads the weights in the internal model

        Parameters
        ----------
        weights : whatever is accepted by the model_to_weights_function
            Weights to be loaded into the model

        Returns
        -------
        None.

        """
        self.weights_to_model_function(self, weights)
        
    def get_weights(self):
        return self.model_to_weights_function(self)
        
    def apply_delta(self, other):
        return self.apply_delta_function(self, other)
    
    def calc_delta(self, other):
        return self.calc_delta_function(self, other)
    
    def apply(self, data):
        return self.apply_model_function(self, data)
    
    def incremental_learn(self, trainingData, trainingOutputs):
        self.incremental_learn_function(self, trainingData, trainingOutputs)
        
    def dump(self, file):
        """
        Dumps the current status of the object, including functions and weights
        
        Parameters
        ----------
        file:
            a file descriptor (open in writable mode)

        Returns
        -------
        Nothing

        """
        outputDict = {
            'model_id': self.model_id,
            'init_model_function': self.init_model_function,
            'apply_model_function': self.apply_model_function,
            'weights_to_model_function': self.weights_to_model_function,
            'model_to_weights_function': self.model_to_weights_function,
            'calc_delta_function': self.calc_delta_function,
            'apply_delta_function': self.apply_delta_function,
            'incremental_learn_function': self.incremental_learn_function,
            'weights': self.get_weights()
            }
        
        dill.dump(outputDict, file)
    
    def dumps(self) -> bytes:
        file = BytesIO()
        self.dump(file)
        return file.getvalue()
    
    def get_empty_copy(self) -> DynamicDLModel:
        """
        Gets an empty copy (i.e. without weights) of the current object

        Returns
        -------
        DynamicDLModel
            Output copy

        """
        return DynamicDLModel(self.model_id, self.init_model_function, self.apply_model_function, self.weights_to_model_function, self.model_to_weights_function, self.calc_delta_function, self.apply_delta_function, self.incremental_learn_function)
    
    @staticmethod
    def Load(file) -> DynamicDLModel:
        """
        Creates an object from a file

        Parameters
        ----------
        file : file descriptor
            A file descriptor.

        Returns
        -------
        DynamicDLModel
            A new instance of a dynamic model

        """
        
        inputDict = dill.load(file)
        outputObj = DynamicDLModel(**inputDict)
        return outputObj
        
    @staticmethod
    def Loads(b: bytes) -> DynamicDLModel:
        """
        Creates an object from a binary dump

        Parameters
        ----------
        file : bytes
            A sequence of bytes

        Returns
        -------
        DynamicDLModel
            A new instance of a dynamic model

        """
        file = BytesIO(b)
        return DynamicDLModel.Load(file)