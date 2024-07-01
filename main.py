import data
import model
import train
import infer
from dataclasses import dataclass

# Model config

@dataclass
class Config():
  # Top Level
  d_model: int = 100
  d_vocab: int = data.vocab_size
  init_range: float = 0.02
  block_size: int = 8

  # LayerNorm
  var_eps: float = 1e-5

  # Attention 
  d_head: int = 64
  n_heads: int = 12

  # MLP
  d_mlp: int = d_model * 4 

  # layers
  n_layers: int = 4


cfg = Config()

model = model.Transformer(cfg)

# Training config

@dataclass
class TrainingConfig():
    batch_size = 16
    steps = 500
    log_every = 10
    lr = 1e-3
    weight_decay = 1e-2

train_cfg = TrainingConfig()

# Train

train.train(model, train_cfg, data.tr_data, data.get_batches)

# Infer

infer.infer(100, model, data.te_data, data.decoder)