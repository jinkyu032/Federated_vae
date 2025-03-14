from models import VAE


__all__ = ['BaseServer']

# Server class for federated learning
class BaseServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.global_model = VAE().to(self.device)

    def aggregate(self, client_weights):
        avg_weights = {key: sum(w[key] for w in client_weights) / len(client_weights) for key in client_weights[0].keys()}
        self.global_model.load_state_dict(avg_weights)


