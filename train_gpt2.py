from dataclasses import dataclass 

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(self.config.vocab_size, self.config.n_emb),
                wpe = nn.Embedding(self.config.block_size, self.config.n_emb),
                h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
                ln_f = nn.LayerNorm(self.config.n_emb)
            )
        )
        self.lm_head = nn.Linear(self.config.n_emb, self.config.vocab_size, bias=False)
        
