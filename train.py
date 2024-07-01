import torch
import torch.nn.functional as f 

# Training loop

def train(model, train_cfg, tr_data, get_batches):
    # Optimizer

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    for i in range(train_cfg.steps):
        tokens, targets = get_batches(tr_data, train_cfg.batch_size, model.cfg.block_size)
        logits = model(tokens)

        # Logits: batch x pos x d_vocab, targets: batch x pos 
        
        loss = f.cross_entropy(logits, f.one_hot(targets, num_classes=model.cfg.d_vocab).float())
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"Step: {(i+1)} , Loss: {(loss)}")
