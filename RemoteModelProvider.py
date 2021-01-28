#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import requests

from .interfaces import ModelProvider
from .DynamicDLModel import DynamicDLModel
from typing import IO, Callable


AVAILABLE_MODELS = ["Classifier", "Thigh", "Leg", "Thigh-Split", "Leg-Split"]



class RemoteModelProvider(ModelProvider):
    
    def __init__(self, models_path, url_base, api_key):
        self.models_path = Path(models_path)
        self.url_base = url_base
        self.api_key = api_key
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
            latest_timestamp = r.json()['latest_timestamp']
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
            print("Model already downloaded. Loading...")
            model = DynamicDLModel.Load(open(latest_model_path, 'rb'))
            return model
        else:
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
            block_size = 1024*1024  # 1 kB
            current_size = 0
            with open(latest_model_path, 'wb') as file:
                for data in r.iter_content(block_size):
                    current_size += len(data)
                    #print(current_size)
                    if progress_callback is not None:
                        progress_callback(current_size, total_size_in_bytes)
                    file.write(data)

            if current_size != total_size_in_bytes:
                print("Download interrupted!")
                os.remove(latest_model_path)
                success = False

        if success:
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
    
    def available_models(self) -> str:
        # TODO: if api_key is invalid return None
        return AVAILABLE_MODELS[:]

    def upload_model(self, modelName: str, model: DynamicDLModel):
        """
        Upload model to server
        
        Args:
            modelName: classifier | thigh | leg
            model: DynamicDLModel
        """
        print("Uploading model...")
        files = {'model_binary': model.dumps()}
        r = requests.post(self.url_base + "upload_model",
                          files=files,
                          data={"model_type": modelName,
                                "api_key": self.api_key})
        print(f"status code: {r.status_code}")
        print(f"message: {r.json()['message']}")

    def _upload_bytes(self, data: IO):
        # TODO implementation of data upload
        # Note: the don't pass data directly to requests because the byte stream is not at the start.
        # Use getbuffer or getvalue instead. See https://github.com/psf/requests/issues/2589
        print("Data upload not yet implemented")