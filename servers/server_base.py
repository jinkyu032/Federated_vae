from models import VAE
from typing import Dict, Union
from omegaconf import DictConfig

__all__ = ['BaseServer']

# Server class for federated learning
class BaseServer:
    def __init__(self, cfg: Union[Dict, DictConfig], model):
        self.cfg = cfg
        self.device = cfg.device
        self.global_model = model

    def aggregate(self, local_outputs):
        client_weights = [w[0] for w in local_outputs]
        avg_weights = {key: sum(w[key] for w in client_weights) / len(client_weights) for key in client_weights[0].keys()}
        self.global_model.load_state_dict(avg_weights)

