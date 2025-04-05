from dataclasses import dataclass

import math
import time
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.adamw

torch.manual_seed(42)
torch.cuda.manual_seed(42)


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
        self.c_proj.NANOGPT_SCALE_INIT = 1
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
        # att = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1))) 
        # att = att.masked_fill(self.bias[:,:,:T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # utilize Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.c_proj.NANOGPT_SCALE_INIT = 1
   
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
    
        # weight sharing scheme; the resulting tensor is gonna be used twice in forward pass, and will add up two branches in the backward pass
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5  # in each transformer block we have two residual path
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"can not forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token embeddings and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layer norm and classifier
        x = self.transformer.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that requires grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layer norms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params ": nondecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)
        print(f"num decay parameters tensors: {len(decay_params)} with {num_decay_params:,} parameters")
        print(f"num nondecay parameters tensors: {len(nondecay_params)} with {num_nondecay_params:,} parameters")
        # create the Adam optimizer and use the fuse version if available
        fused_available = 'fused' in inspect.signature(torch.optim.Adamw).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=[0.9, 0.95],
            eps=1e-8,
            fused=use_fused,
            weight_decay=weight_decay,
        )
        return optimizer
        

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        with open('dataset/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_positin = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_positin: self.current_positin + (B * T) + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)
        # advance the position in the tensor
        self.current_positin += B * T
        # if loading next batch would be out of bounds, reset
        if self.current_positin + (B * T + 1) > len(self.tokens):
            self.current_positin = 0

        return x, y


num_return_sequences = 5
max_length = 30

device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"
# print(f"Your device is: {device}")


model = GPT(GPTConfig(vocab_size=50304))  # we have preferences for numbers which are power of two!
model.to(device)
model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min_learning_rate 
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio < 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0 
    return min_lr + coeff * (max_lr - min_lr)


train_loader = DataLoaderLite(B=4, T=32)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=[0.9, 0.95], eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)


for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 

    optimizer.step()
    # torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_second = tokens_processed / dt

    print(f"step {step} | loss {loss.item()} | lr: {lr:.4f} | norm: {norm:.4f} | dt: {dt:.4f}")


# while x.size(1) < max_length:
#     # forward the model to get the logits
#     with torch.no_grad():
#         logits = model(x)  # (B, T, vocab_size)
#         # take the logits at the last position
#         logits = logits[:, -1, :]    # (B, vocab_size)
#         # get the probabilities
#         probs = F.softmax(logits, dim=-1)
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select a token from the top-k probabilities
#         ix = torch.multinomial(topk_probs, 1)  # (B, 1)
#         # gather the corresponding indices 
#         xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim=1)


# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(f"> {decoded}")
