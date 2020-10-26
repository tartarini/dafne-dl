#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:39:39 2020

@author: francesco
"""
from .interfaces import ModelProvider
from .DynamicDLModel import DynamicDLModel
import os

MODEL_NAMES_MAP = {
    'Classifier': 'classifier.model',
    'Thigh': 'thigh.model',
    'Leg': 'leg.model'
    }

class LocalModelProvider(ModelProvider):
    
    def __init__(self, models_path):
        self.models_path = models_path
        
    def load_model(self, modelName: str) -> DynamicDLModel:
        print("Loading model:", modelName)
        model_file = os.path.join(self.models_path, MODEL_NAMES_MAP[modelName])
        print("Done")
        return DynamicDLModel.Load( open(model_file, 'rb'))
    
    def available_models(self) -> str:
        return list(MODEL_NAMES_MAP.keys())