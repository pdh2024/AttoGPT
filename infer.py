import torch
import einops as e
import data
import model
import train

def infer(n, model, te_data, decoder):
    rand = torch.randint(0, len(te_data)-model.cfg.block_size,(1,))
    tokens = te_data[rand:rand+model.cfg.block_size]

    seq = tokens
    print(decoder(seq))

    for i in range(n):
        tokens = te_data[rand+i:rand+i+model.cfg.block_size]
        tokens = e.rearrange(tokens, '(batch pos) -> batch pos', batch=1)
        logits = model(tokens)

        logits = e.rearrange(logits, 'batch pos d_vocab -> (batch pos) d_vocab')
        probs = logits.softmax(dim=-1)

        new_token = torch.argmax(probs[-1])

        seq = torch.concat((seq, new_token.unsqueeze(dim=-1)), dim=0)
  
    print(decoder(seq))
