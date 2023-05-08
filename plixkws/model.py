
import os
import wget
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from . import backbone as bk
from . import protonet as pt
from . import util 

def load(encoder_name: str = "base", language: str = "multi", 
        models_dir: str = "models", device: str = "cuda"):

    if encoder_name == "base":
        if language not in ["multi", "en"]:
            raise ValueError("`base' encoder only available for `en` and `multi` models.")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    models_dir = Path(os.path.join(models_dir))
    models_dir.mkdir(exist_ok=True)

    config_file_path = os.path.join(models_dir, "config.json")
    if not os.path.exists(config_file_path):
        config_url = "https://www.dropbox.com/s/ipmytoirguvzg2u/config.json?dl=1"
        wget.download(config_url,  out=config_file_path)
    with open(config_file_path) as f:
        config = json.load(f)

    model_weights = os.path.join(models_dir, f"{encoder_name}_{language}_model.pt")
    model_weights_url = config["urls"][f"{encoder_name}_{language}"]
    if not os.path.exists(model_weights):
        wget.download(model_weights_url, out=model_weights)

    model = pt.ProtoNet(bk.Backbone(encoder_name=encoder_name))
    checkpoint = torch.load(model_weights, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    return model

