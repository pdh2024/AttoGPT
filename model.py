import torch
import torch.nn as nn
import einops as e
import math

# Embedding

class Embed(nn.Module):
    def __init__(self, cfg):
      super().__init__()
      self.cfg = cfg

      self.w_e = nn.Parameter(torch.empty(self.cfg.d_vocab, self.cfg.d_model))
      nn.init.normal_(self.w_e, std=self.cfg.init_range)

    def forward(self, tokens):
      # Tokens: batch x pos 

      embed = self.w_e[tokens, :]
      return embed

# Positional embedding

class PosEmbed(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

    self.w_pos = nn.Parameter(torch.empty((self.cfg.block_size, self.cfg.d_model)))
    nn.init.normal_(self.w_pos, std=self.cfg.init_range)

  def forward(self, tokens):
    # Tokens: batch x pos

    pos_embed = self.w_pos[0:tokens.size(-1), :]
    pos_embed = e.repeat(pos_embed, 'pos d_model -> batch pos d_model', batch=tokens.size(0))
    return pos_embed

# LayerNorm

class LayerNorm(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

    self.w = nn.Parameter(torch.ones(self.cfg.d_model))
    nn.init.normal_(self.w, std=self.cfg.init_range)

    self.b = nn.Parameter(torch.zeros(self.cfg.d_model))

  def forward(self, resid):
     # Resid: batch x pos x d_model

     # Subtract mean
     mean = e.reduce(resid, 'batch pos d_model -> batch pos ()', 'mean')
     resid = resid - mean

     # Divide by square root of var
     var = e.reduce(resid.pow(2), 'batch pos d_model -> batch pos ()', 'mean')
     resid = resid/(torch.sqrt(var+self.cfg.var_eps))

     # Scale and shift
     resid = resid*self.w+self.b

     return resid

# Attention
    
class Attention(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

    self.w_q = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head))
    nn.init.normal_(self.w_q, std=self.cfg.init_range)

    self.w_k = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head))
    nn.init.normal_(self.w_k, std=self.cfg.init_range)

    self.w_v = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head))
    nn.init.normal_(self.w_v, std=self.cfg.init_range)

    self.w_o = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model))
    nn.init.normal_(self.w_o, std=self.cfg.init_range)

  def forward(self, resid):
    # Resid: batch x pos x d_model

    # Get query, key, and value matrices

    qs = e.einsum(resid, self.w_q, 'batch pos d_model,n_heads d_model d_head->batch n_heads pos d_head')
    ks = e.einsum(resid, self.w_k, 'batch pos d_model,n_heads d_model d_head->batch n_heads pos d_head')
    vs = e.einsum(resid, self.w_v, 'batch pos d_model,n_heads d_model d_head->batch n_heads pos d_head')

    # Get attention pattern

    attn = e.einsum(qs, ks, 'batch n_heads query_pos d_head, batch n_heads key_pos d_head->batch n_heads query_pos key_pos')
    attn.masked_fill_(self.get_causal_mask(attn), float('-inf'))
    attn = (attn/math.sqrt(self.cfg.d_head)).softmax(dim=-1)

    # Weigh value vectors according to pattern

    z = e.einsum(attn, vs, 'batch n_heads query_pos key_pos, batch n_heads key_pos d_head->batch n_heads query_pos d_head')

    out = e.einsum(z,self.w_o,'batch n_heads pos d_head,n_heads d_head d_model->batch pos d_model')

    return out
  
  def get_causal_mask(self, attn):
    return torch.triu(torch.ones(attn.size(-2), attn.size(-1), device=attn.device), diagonal=1).bool()

# MLP

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MLP(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

    self.w_in = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp))
    nn.init.normal_(self.w_in, std=self.cfg.init_range)
    self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp))

    self.w_out = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model))
    nn.init.normal_(self.w_out, std=self.cfg.init_range)
    self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model))
  
  def forward(self, resid):
    # Resid: batch x pos x d_model
    
    resid = e.einsum(resid, self.w_in,'batch pos d_model, d_model d_mlp->batch pos d_mlp')+self.b_in
    resid = new_gelu(resid)
    resid = e.einsum(resid, self.w_out, 'batch pos d_mlp, d_mlp d_model->batch pos d_model')+self.b_out
    return resid

# Block

class Block(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg=cfg
    
    self.ln1 = LayerNorm(cfg)
    self.ln2 = LayerNorm(cfg)

    self.attn = Attention(cfg)
    self.mlp = MLP(cfg)

  def forward(self, resid):
    # Resid: batch x pos x d_model

    resid = resid + self.attn(self.ln1(resid))
    resid = resid + self.mlp(self.ln2(resid))

    return resid

# Unembedding

class Unembed(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

    self.w_u = nn.Parameter(torch.empty((self.cfg.d_model, self.cfg.d_vocab)))
    nn.init.normal_(self.w_u, std=self.cfg.init_range)

  def forward(self, resid):
    # Resid: batch x pos x d_model

    logits = e.einsum(resid, self.w_u, 'batch pos d_model, d_model d_vocab->batch pos d_vocab')
    return logits

# Model

class Transformer(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

    self.embed = Embed(cfg)
    self.pos_embed = PosEmbed(cfg)
    self.ln_final = LayerNorm(cfg)
    self.unembed = Unembed(cfg)

    self.blocks = nn.ModuleList([Block(cfg) for i in range(self.cfg.n_layers)])

  def forward(self, tokens):
    # Tokens: batch x pos
    resid = self.embed(tokens)+self.pos_embed(tokens)

    for block in self.blocks:
      resid = block(resid)

    resid_norm = self.ln_final(resid)
    logits = self.unembed(resid_norm)

    # Logits: batch x pos x d_vocab
    return logits
