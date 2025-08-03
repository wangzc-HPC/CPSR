import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datasets import load_dataset
import random
import numpy as np
import os
import time
from collections import OrderedDict
from tqdm import *
import gc

import sys

sys.path.append("/lihongliang/wangzc/")
from GPT.pipeline.Gpipe.GPT2 import *
from GPT import PreRecover_ck


# Initialize DDP
def init_ddp(local_rank, world_size):
    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)  # Correct → each node uses cuda:0


# Get current rank (needed when modifying main function)
rank = int(os.environ['LOCAL_RANK'])  # Always 0 per node (since each node has only 1 GPU)
print("local rank is {}".format(rank))
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
print("world_size is {}".format(world_size))

init_ddp(rank, world_size)

torch.manual_seed(12046)

# GPT-2 medium configuration
emb_size = 1024
head_size = 64
n_layer = 24
sequence_len = 256
learning_rate = 1e-4
eval_iters = 20
batch_size = 12
epochs = 10


def attention(query, key, value, dropout, mask=None):
    # query, key, value all have same shape
    B, T, C = query.shape
    # (B, T, C) @ (B, C, T) --> (B, T, T)
    scores = query @ key.transpose(-2, -1) / (C ** 0.5)
    if mask is not None:
        # If no mask, tokens can use context from both sides (bidirectional attention)
        # If mask is upper triangular matrix, it's unidirectional autoregressive attention
        # Mask shape is (T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    w_att = dropout(F.softmax(scores, dim=-1))  # (B, T, T)
    out = w_att @ value  # (B, T, C)
    return out, w_att


class MaskedAttention(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        # This upper triangular matrix doesn't participate in model training
        self.register_buffer(
            'tril', torch.tril(torch.ones(sequence_len, sequence_len)))
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)
        mask = self.tril[:T, :T]
        out, _ = attention(q, k, v, self.dropout, mask)
        return out  # (B, T, H)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        # Ensure feature length is multiple of context vector length
        assert (emb_size % head_size == 0)
        # Define number of single-head attentions
        n_head = emb_size // head_size
        heads = [MaskedAttention(emb_size, head_size) for _ in range(n_head)]
        self.heads = nn.ModuleList(heads)
        # Linear transformation
        self.proj = nn.Linear(emb_size, emb_size)
        # Dropout
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Concatenate results from multiple single-head attentions
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.l1 = nn.Linear(emb_size, 4 * emb_size)
        self.l2 = nn.Linear(4 * emb_size, emb_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.gelu(self.l1(x))
        out = self.dropout(self.l2(x))
        return out


class Block(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        self.mha = MaskedMultiHeadAttention(emb_size, head_size)
        self.ff = FeedForward(emb_size)
        # Layer normalization
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # Residual connection
        x = x + self.mha(self.ln1(x))  # (B, T, C)
        out = x + self.ff(self.ln2(x))  # (B, T, C)
        return out


class CharGPT(nn.Module):
    def __init__(self, vs):
        super().__init__()
        # Token embedding layer
        self.token_embedding = nn.Embedding(vs, emb_size)
        # Position embedding layer
        self.position_embedding = nn.Embedding(sequence_len, emb_size)
        # Decoder blocks
        blocks = [Block(emb_size, head_size) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*blocks)
        self.ln = nn.LayerNorm(emb_size)
        # Language modeling head
        self.lm_head = nn.Linear(emb_size, vs)

    def forward(self, x):
        B, T = x.shape
        # Define token positions with shape (T)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        # Token semantic features
        tok_emb = self.token_embedding(x)  # (B, T,  C)
        # Position features
        pos_emb = self.position_embedding(pos)  # (   T,  C)
        x = tok_emb + pos_emb  # (B, T,  C)
        x = self.blocks(x)  # (B, T,  C)
        x = self.ln(x)  # (B, T,  C)
        logits = self.lm_head(x)  # (B, T, vs)
        return logits


class char_tokenizer:
    def __init__(self, data):
        # All characters appearing in data form the dictionary
        chars = sorted(list(set(''.join(data))))
        # Reserve a position for the special end token
        self.char2ind = {s: i + 1 for i, s in enumerate(chars)}
        self.char2ind['<|e|>'] = 0
        self.ind2char = {i: s for s, i in self.char2ind.items()}

    def encode(self, text):
        return [self.char2ind[c] for c in text]

    def decode(self, enc):
        if isinstance(enc, int):
            return self.ind2char[enc]
        return [self.ind2char[i] for i in enc]


# Dataset processing
raw_datasets = load_dataset(
    path="/lihongliang/wangzc/GPT/dataset/CodeSearchNet/code_search_net.py",
    name="python",
    data_dir="/lihongliang/wangzc/GPT/dataset/CodeSearchNet"  # Passed to builder
)
datasets = raw_datasets['train'].filter(lambda x: 'apache/spark' in x['repository_name'])
tok = char_tokenizer(datasets['whole_func_string'])

# Initialize model (with DDP wrapper)
model = CharGPT(len(tok.char2ind))
model = model.to(rank)
model = DDP(model, device_ids=[rank])
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def generate_batch(Stage_list, idx, max_new_tokens=300):
    '''
    Generate text using model (repeatedly using model for prediction)
    Parameters
    ----
    model  CharGPT  Text generation model
    idx  torch.LongTensor  Current character position in dictionary with shape (1, T)
    max_new_tokens  int  Maximum length of generated text
    Returns
    ----
    out  list[int]  Generated text
    '''
    # Switch model to evaluation mode
    for stage in Stage_list:
        stage.eval()
    for _ in range(max_new_tokens):
        # Limit context length to avoid errors
        logits = idx[:, -sequence_len:]
        # Text generation is computationally inefficient due to repeated calculations
        for stage in Stage_list:
            logits = stage.forward(logits)
        # Only use last prediction result
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # Get final prediction (next character) based on model probabilities
        # This step has some randomness
        ix = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, ix), dim=1)
        if ix.item() == 0:
            break
    # Switch model back to training mode
    for stage in Stage_list:
        stage.train()
    return idx.tolist()[0]


def process(data, sequence_len=256):
    '''
    Generate training data from text
    '''
    # text is list of strings
    text = data['whole_func_string']
    inputs, labels = [], []
    for i in text:
        enc = tok.encode(i)
        # 0 corresponds to text end
        enc += [0]
        # Convert text to multiple training samples
        for i in range(len(enc) - sequence_len):
            inputs.append(enc[i: i + sequence_len])
            # Prediction label is next character, so just shift by one position
            labels.append(enc[i + 1: i + 1 + sequence_len])
    return {'inputs': inputs, 'labels': labels}


# Split data into training and test sets
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
# Convert text to training data containing inputs and labels
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
tokenized.set_format(type='torch')

print(tokenized['train']['inputs'].shape, tokenized['train']['labels'].shape)

# Build data loaders
train_sampler = DistributedSampler(
    tokenized['train'],
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)
train_loader = DataLoader(
    tokenized['train'],
    batch_size=batch_size,
    sampler=train_sampler,
    pin_memory=True
)
test_loader = DataLoader(
    tokenized['test'],
    batch_size=batch_size,
    shuffle=False
)

print("training step num is {}".format(len(train_loader)))
print("test step num is {}".format(len(test_loader)))


def estimate_loss(model):
    re = {}
    # Switch model to evaluation mode
    model.eval()
    re['train'] = _loss(model, train_loader)
    re['test'] = _loss(model, test_loader)
    # Switch model back to training mode
    model.train()
    return re


@torch.no_grad()
def _loss(model, data_loader):
    '''
    Calculate model evaluation metrics on different datasets
    '''
    loss = []
    data_iter = iter(data_loader)
    # Use multiple random batches to estimate model performance
    for k in range(eval_iters):
        data = next(data_iter, None)
        if data is None:
            data_iter = iter(data_loader)
            data = next(data_iter, None)
        inputs, labels = data['inputs'].to(rank), data['labels'].to(rank)
        logits = model(inputs)
        # According to cross_entropy definition, need to transpose logits
        # See official cross_entropy documentation for details
        logits = logits.transpose(-2, -1)
        loss.append(F.cross_entropy(logits, labels).item())
    return torch.tensor(loss).mean().item()


def save_partial_model(model, path, num_layers_to_save):
    # 1. Save model parameters (only keep first 6 encoder layers)
    model_state = model.state_dict()
    partial_state = OrderedDict()

    for key in model_state:
        # Filter parameters belonging to first 6 encoder layers
        if 'module.blocks.' in key:
            layer_num = int(key.split('.')[2])  # Assume parameter name format encoder.layers.0.xxx
            if layer_num < num_layers_to_save:
                partial_state[key] = model_state[key]

    # 3. Save checkpoint
    torch.save({
        'model_state': partial_state,
    }, path)
    print(f"Saved first {num_layers_to_save} layer checkpoint to {path}")


group = dist.new_group([0, 1, 2, 3])
global_rank = int(os.environ["RANK"])


def train_gpt(model, optimizer, data_loader, recovery, e, step, epochs=10):
    lossi = []
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Important: ensure different shuffle each epoch
        if recovery and epoch < e:
            continue
        iter_start = time.time()
        for i, data in tqdm(enumerate(data_loader, 0)):
            if recovery and epoch == e and i <= step:
                continue

            inputs, labels = data['inputs'].to(rank), data['labels'].to(rank)
            optimizer.zero_grad()

            forward_start = time.time()
            logits = model(inputs)
            # According to cross_entropy definition, need to transpose logits
            logits = logits.transpose(-2, -1)
            loss = F.cross_entropy(logits, labels)
            forward_end = time.time()
            print(f"forward time is {forward_end - forward_start}s")

            lossi.append(loss.item())

            backward_start = time.time()
            loss.backward()
            backward_end = time.time()
            print(f"backward time is {backward_end - backward_start}s")

            allreduce_start = time.time()
            for p in model.parameters():
                p.grad = p.grad / world_size
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group, async_op=False)
            allreduce_end = time.time()

            optimizer.step()
            if i % 50 == 0:
                stats = estimate_loss(model)
                print(stats)

            if i % 1 == 0:
                print('epoch is {}, iter is {}, loss is {}'.format(epoch, i, loss))
    return lossi


# Checkpoint recovery
epoch = 0
i = 0
# recovery = True
recovery = False
if recovery:
    ck_path = "/lihongliang/wangzc/GPT/dp/checkpoint/gpt2_medium/1n1g/epoch_2_iter_10000_rank_0.pth"
    ck = torch.load(ck_path)

    model.load_state_dict(ck['model_state_dict'])
    optimizer.load_state_dict(ck['optimizer_state_dict'])

    # Restore random states
    seeds = ck['seeds']
    random.setstate(seeds[0])
    np.random.set_state(seeds[1])

    if seeds[2].device != torch.device('cpu'):
        # If RNG state not on CPU, move to CPU
        seeds[2] = seeds[2].cpu()
    torch.set_rng_state(seeds[2])
    torch.cuda.set_rng_state_all(seeds[3])

    epoch = ck['epoch']
    i = ck['iteration']

    del ck  # Delete reference
    torch.cuda.empty_cache()  # Clear PyTorch CUDA cache
    gc.collect()  # Trigger Python garbage collection

l = train_gpt(model, optimizer, train_loader, recovery, epoch, i)