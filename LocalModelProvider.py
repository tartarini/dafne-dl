#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:39:39 2020

@author: francesco
"""
from pathlib import Path

from .interfaces import ModelProvider
from .DynamicDLModel import DynamicDLModel


AVAILABLE_MODELS = ["Classifier", "Thigh", "Leg"]

class LocalModelProvider(ModelProvider):
    
    def __init__(self, models_path):
        self.models_path = Path(models_path)
        
    def load_model(self, modelName: str) -> DynamicDLModel:
        print(f"Loading model: {modelName}")
        model_file = list(self.models_path.glob(f"{modelName}_*.model"))
        if len(model_file) == 0:
            raise FileNotFoundError("Could not find model file.")
        if len(model_file) > 1:
            raise ValueError(f"More than one '{modelName}' model found.")
        return DynamicDLModel.Load(open(model_file[0], 'rb'))
    
    def available_models(self) -> str:
        return AVAILABLE_MODELS

    def upload_model(self, modelName: str, model: DynamicDLModel):
        print("You are using the LocalModelProvider. Therefore no upload is done!")