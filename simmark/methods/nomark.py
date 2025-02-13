import torch

class NoMarkProcessor(torch.nn.Module):
    def __init__(self, gen_config):
        super().__init__()
    
    def forward(self, input_ids, logits):
        return logits