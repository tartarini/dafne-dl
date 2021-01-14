#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import requests

from .interfaces import ModelProvider
from .DynamicDLModel import DynamicDLModel


AVAILABLE_MODELS = ["Classifier", "Thigh", "Leg"]


class RemoteModelProvider(ModelProvider):
    
    def __init__(self, models_path):
        self.models_path = Path(models_path)

        # todo: put this into a config file
        self.url_base = 'http://localhost:5000/'
        self.api_key = "abc123"

    def load_model(self, modelName: str) -> DynamicDLModel:
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
            print(f"message: {r.json()['message']}")
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
                                "api_key": self.api_key})
        if r.ok:
            model = DynamicDLModel.Loads(r.content)
            model.dump(open(latest_model_path, "wb"))
            return model
        else:
            print("ERROR: Request to server failed")
            print(f"status code: {r.status_code}")
            print(f"message: {r.json()['message']}")
            return None
    
    def available_models(self) -> str:
        return AVAILABLE_MODELS

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
