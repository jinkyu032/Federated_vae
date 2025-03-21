from .server_base import BaseServer
from typing import Dict, Union
from omegaconf import DictConfig



def get_server(cfg: Union[Dict, DictConfig], model):
    if isinstance(cfg, DictConfig):
        server_type = cfg.server.server_type
    else:
        server_type = cfg.server_type
        
    if server_type == "base":
        return BaseServer(cfg, model)
    else:
        raise ValueError(f"Invalid server type: {server_type}")
