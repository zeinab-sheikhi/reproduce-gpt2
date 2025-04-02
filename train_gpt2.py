from dataclasses import dataclass

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_emb: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        # key, value, query projections for all heads, but in a batch 
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)
        # output projection
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        # regularization 
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        # not really a bias, more of a mask, but following the OpenAI-HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimention
        # calculate query, key, value for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) is nh * hs
        # e.g. in GPT2 (124M), n_head=12, hs=64, so we have C = 12 * 64 = 768 channels in the transformer.
        qkv = self.c_attn(x)  # (B, T, 3 * C) concatenated q, k, v
        q, k, v = qkv.split(self.n_emb, dim=2)  # (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # attention materializes the large (T, T) matrix for all queries and keys
        att = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1))) 
        att = att.masked_fill(self.bias[:,:,:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # reassemble all heads outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
        

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_emb * 4, config.n_emb)
   
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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
      
    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT2 model weights from HuggingFace."""
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head, and n_emb are determined from model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_emb=768),   # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_emb=1024),  # 350M params  
            'gpt2-large':  dict(n_layer=36, n_head=20, n_emb=1280),  # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_emb=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024   # always 1024 for GPT model checkpoints
        
        # create a from-scratch initialized miniGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]  # discard this mask / buffer (they are not parameters)

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while making sure all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


model = GPT.from_pretrained("gpt2")
print('did not crash, yay!')
