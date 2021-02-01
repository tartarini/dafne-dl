#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
        self.upload_dir = upload_dir

    def get_model_names(self):
        model_list = self.models_path.glob('*.model')
        model_names = list(set([os.path.basename(s).split('_')[0] for s in model_list])) # get the name of the model, which is the part of the file before the '_'
        model_names = list(filter(None, model_names)) # remove any empty names
        return model_names

    def load_model(self, modelName: str, progress_callback: Callable[[int, int], None] = None) -> DynamicDLModel:
        print(f"Loading model: {modelName}")
        model_file = sorted(list(self.models_path.glob(f"{modelName}_*.model")))
        if len(model_file) == 0:
            raise FileNotFoundError("Could not find model file.")
        print('Opening', model_file[-1])
        return DynamicDLModel.Load(open(model_file[-1], 'rb'))

    def import_model(self, file_path, model_name):
        print('Importing model')
        model = DynamicDLModel.Load(open(file_path, 'rb'))
        self.upload_model(model_name, model)

    def available_models(self) -> str:
        return self.get_model_names()

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