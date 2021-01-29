#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:39:39 2020

@author: francesco
"""
from pathlib import Path

from .interfaces import ModelProvider
from .DynamicDLModel import DynamicDLModel
from typing import Union, IO
import os
import datetime
from typing import Callable


AVAILABLE_MODELS = ["Classifier", "Thigh", "Leg", "Thigh-Split", "Leg-Split"]
OUTPUT_DATA_DIR = 'data_out'


class LocalModelProvider(ModelProvider):
    
    def __init__(self, models_path):
        self.models_path = Path(models_path)
        
    def load_model(self, modelName: str, progress_callback: Callable[[int, int], None] = None) -> DynamicDLModel:
        print(f"Loading model: {modelName}")
        model_file = list(self.models_path.glob(f"{modelName}_*.model"))
        if len(model_file) == 0:
            raise FileNotFoundError("Could not find model file.")
        if len(model_file) > 1:
            raise ValueError(f"More than one '{modelName}' model found.")
        return DynamicDLModel.Load(open(model_file[0], 'rb'))
    
    def available_models(self) -> str:
        return AVAILABLE_MODELS[:]

    def upload_model(self, modelName: str, model: DynamicDLModel):
        print("You are using the LocalModelProvider. Therefore no upload is done!")

    def _upload_bytes(self, data: IO):
        print("You are using the LocalModelProvider. Therefore no upload is done!")
        os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
        filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.npz")
        with open(os.path.join(OUTPUT_DATA_DIR, filename), 'wb') as f:
            f.write(data.getbuffer())
        print('File saved')