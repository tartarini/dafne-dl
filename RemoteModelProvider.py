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

import os
from pathlib import Path
import requests

from .interfaces import ModelProvider
from .DynamicDLModel import DynamicDLModel
from typing import IO, Callable, List, Union
import threading
import time
import datetime
from .misc import calculate_file_hash

UPLOAD_RETRIES = 3
TIME_BETWEEN_RETRIES = 10


def upload_model(url_base, filename, modelName, api_key, dice):
    print('Calculating hash...')
    file_hash = calculate_file_hash(filename)
    print(file_hash)
    for retries in range(UPLOAD_RETRIES):
        print(f"Sending {filename}")
        files = {'model_binary': open(filename, 'rb')}
        r = requests.post(url_base + "upload_model",
                          files=files,
                          data={"model_type": modelName,
                                "api_key": api_key,
                                "dice": dice,
                                "hash": file_hash})
        print(f"status code: {r.status_code}")
        try:
            print(f"message: {r.json()['message']}")
        except:
            pass

        if r.status_code == 200:
            print("upload successful") # success
            break
        print('Upload error')
        time.sleep(TIME_BETWEEN_RETRIES)
    os.remove(filename)


def upload_data(url_base, filename, api_key):
    for retries in range(UPLOAD_RETRIES):
        print(f"Sending {filename}")
        files = {'data_binary': open(filename, 'rb')}
        r = requests.post(url_base + "upload_data",
                          files=files,
                          data={"api_key": api_key})
        print(f"status code: {r.status_code}")
        try:
            print(f"message: {r.json()['message']}")
        except:
            pass

        if r.status_code == 200:
            print("upload successful") # success
            break
        print('Upload error')
        time.sleep(TIME_BETWEEN_RETRIES)
    os.remove(filename)


class RemoteModelProvider(ModelProvider):
    
    def __init__(self, models_path, url_base, api_key, temp_upload_dir):
        self.models_path = Path(models_path)
        self.url_base = url_base
        self.api_key = api_key
        self.temp_upload_dir = temp_upload_dir
        os.makedirs(self.models_path, exist_ok=True)
        print(f"Config: {self.url_base}, {self.api_key}")

    def load_model(self, modelName: str, progress_callback: Callable[[int, int], None] = None) -> DynamicDLModel:
        """
        Load latest model from remote server if it does not already exist locally.

        Args:
            modelName: Classifier | Thigh | Leg

        Returns:
            DynamicDLModel or None
        """
        print(f"Loading model: {modelName}")

        # Get the name of the latest model
        r = requests.post(self.url_base + "info_model",
                          json={"model_type": modelName,
                                "api_key": self.api_key})
        if r.ok:
            json_content = r.json()
            latest_timestamp = json_content['latest_timestamp']
            file_hash_remote = json_content['hash']
        else:
            print("ERROR: Request to server failed")
            print(f"status code: {r.status_code}")
            try:
                print(f"message: {r.json()['message']}")
            except:
                pass
            return None

        # Check if model already exists locally
        latest_model_path = self.models_path / f"{modelName}_{latest_timestamp}.model"
        if os.path.exists(latest_model_path):
            print("Model already downloaded. Checking hash...")
            file_hash_local = calculate_file_hash(latest_model_path)
            if file_hash_local == file_hash_remote:
                print('Model exists, skipping download')
                model = DynamicDLModel.Load(open(latest_model_path, 'rb'))
                return model
            else:
                print('Local model is corrupt')
                os.remove(latest_model_path)

        print("Downloading new model...")

        # Receive model
        r = requests.post(self.url_base + "get_model",
                          json={"model_type": modelName,
                                "timestamp": latest_timestamp,
                                "api_key": self.api_key},
                          stream=True)
        success = False
        if r.ok:
            success = True
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            print("Size to download:", total_size_in_bytes)
            block_size = 1024*1024  # 1 MB
            current_size = 0
            with open(latest_model_path, 'wb') as file:
                for data in r.iter_content(block_size):
                    current_size += len(data)
                    #print(current_size)
                    if progress_callback is not None:
                        progress_callback(current_size, total_size_in_bytes)
                    file.write(data)

            print("Downloaded size", current_size)
            file_hash_local = calculate_file_hash(latest_model_path)

            if current_size != total_size_in_bytes or file_hash_local != file_hash_remote:
                print("Download failed!")
                os.remove(latest_model_path)
                success = False

        if success:
            print('Model check OK')
            model = DynamicDLModel.Load(open(latest_model_path, "rb"))

            # Deleting older models
            old_models = self.models_path.glob(f"{modelName}_*.model")
            print("Deleting old models: ")
            for old_model in old_models:
                if old_model != latest_model_path:
                    print(f"  Deleting: {str(old_model)}")
                    os.remove(old_model)

            return model
        else:
            print("ERROR: Request to server failed")
            print(f"status code: {r.status_code}")
            try:
                print(f"message: {r.json()['message']}")
            except:
                pass
            return None
    
    def available_models(self) -> Union[None, List[str]]:
        r = requests.post(self.url_base + "get_available_models",
                          json={"api_key": self.api_key})
        if r.ok:
            models = r.json()['models']
            return models
        else:
            print(f"status code: {r.status_code}")
            try:
                print(f"message: {r.json()['message']}")
            except:
                pass
            if r.status_code == 401:
                raise PermissionError("Your api_key is invalid.")
            return None

    def upload_model(self, modelName: str, model: DynamicDLModel, dice_score: float = 0.0):
        """
        Upload model to server
        
        Args:
            modelName: classifier | thigh | leg
            model: DynamicDLModel
        """
        print("Uploading model...")
        filename_out = os.path.join(self.temp_upload_dir, f'{modelName}_{model.timestamp_id}.model')
        model.dump(open(filename_out, 'wb'))
        upload_thread = threading.Thread(target=upload_model, args=(self.url_base, filename_out, modelName,
                                                                            self.api_key, dice_score))
        upload_thread.start()

    def _upload_bytes(self, data: IO):
        # Note: the don't pass data directly to requests because the byte stream is not at the start.
        # Use getbuffer or getvalue instead. See https://github.com/psf/requests/issues/2589
        print("Uploading data")
        filename = datetime.datetime.now().strftime("data_%Y%m%d_%H%M%S.npz")
        filename_out = os.path.join(self.temp_upload_dir, filename)
        with open(filename_out, 'wb') as f:
            f.write(data.getbuffer())
        upload_thread = threading.Thread(target=upload_data, args=(self.url_base, filename_out, self.api_key))
        upload_thread.start()
