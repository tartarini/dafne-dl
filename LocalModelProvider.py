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

class LocalModelProvider(ModelProvider):
    
    def __init__(self, models_path, upload_dir):
        self.models_path = Path(models_path)
        model_list = self.models_path.glob('*.model')
        model_names = list(set([os.path.basename(s).split('_')[0] for s in model_list])) # get the name of the model, which is the part of the file before the '_'
        self.model_names = list(filter(None, model_names)) # remove any empty names
        self.upload_dir = upload_dir

    def load_model(self, modelName: str, progress_callback: Callable[[int, int], None] = None) -> DynamicDLModel:
        print(f"Loading model: {modelName}")
        model_file = sorted(list(self.models_path.glob(f"{modelName}_*.model")))
        if len(model_file) == 0:
            raise FileNotFoundError("Could not find model file.")
        print('Opening', model_file[-1])
        return DynamicDLModel.Load(open(model_file[-1], 'rb'))
    
    def available_models(self) -> str:
        return self.model_names[:]

    def upload_model(self, modelName: str, model: DynamicDLModel, dice_score: float=0.0):
        print("You are using the LocalModelProvider. Model is saved in the model directory!")
        filename = f'{modelName}_{model.timestamp_id}.model'
        print('Saving', filename)
        model.dump(open(os.path.join(self.models_path, filename), 'wb'))

    def _upload_bytes(self, data: IO):
        print("You are using the LocalModelProvider. Therefore no upload is done!")
        filename = datetime.datetime.now().strftime("data_%Y%m%d_%H%M%S.npz")
        with open(os.path.join(self.upload_dir, filename), 'wb') as f:
            f.write(data.getbuffer())
        print('File saved')