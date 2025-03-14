from .server_base import BaseServer




def get_server(cfg):
    if cfg.server_type == "base":
        return BaseServer(cfg)
    else:
        raise ValueError(f"Invalid server type: {cfg.server_type}")