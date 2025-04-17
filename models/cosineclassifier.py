import torch
import torch.nn as nn
from utils.data import idx2onehot

class cosine_classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cosine_classifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)



        

    def forward(self, x):
        #x = x.normalize(p=2, dim=1)
        x = nn.functional.normalize(x, p=2, dim=1)
        linear_weight = self.linear.weight
        norm_linear_weight = nn.functional.normalize(linear_weight, p=2, dim=1)
        cosine_similarity = torch.mm(x, norm_linear_weight.t())
        return cosine_similarity
