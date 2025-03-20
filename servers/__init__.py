from .server_base import BaseServer




def get_server(cfg, model):
    if cfg.server_type == "base":
        return BaseServer(cfg, model)
    else:
        raise ValueError(f"Invalid server type: {cfg.server_type}")