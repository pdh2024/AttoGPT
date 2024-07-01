import torch

# Synthetic data 

text = ''
for i in range(10000):
  text = text + str(i % 10)

# Get vocab

vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# Map to tokens

char_to_int = {c:i for i,c in list(enumerate(vocab))}
int_to_char = {char_to_int[c]:c for c in char_to_int.keys()}

encoder = lambda s: [char_to_int[c] for c in s]
decoder = lambda l: ''.join([int_to_char[int(i)] for i in l])

# Create datasets

data = torch.tensor(encoder(text))
n = int(0.9*len(data))

te_data = data[:n]
tr_data = data[n:]

# Get batches of inputs and targets

torch.manual_seed(1337)

def get_batches(data, batch_size, block_size):
  rands = torch.randint(0, len(data)-block_size, (batch_size, ))

  xs = torch.stack([data[r:r+block_size] for r in rands])
  ys = torch.stack([data[r+1:r+block_size+1] for r in rands])

  return xs, ys