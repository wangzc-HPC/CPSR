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

# 初始化DDP
def init_ddp(local_rank, world_size):
    # os.environ['MASTER_ADDR'] = '172.22.153.25'
    # os.environ['MASTER_PORT'] = '1231'
    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)  # 正确→每个节点都用cuda:0
    # torch.cuda.set_device(rank)

# 获取当前rank（修改主函数时需要）
# rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
rank = int(os.environ['LOCAL_RANK'])  # 每节点总是0（因为每节点只有1个GPU）
print("local rank is {}".format(rank))
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
print("world_size is {}".format(world_size))

init_ddp(rank, world_size)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(12046)

# # GPT-3.35B
# emb_size = 4096
# head_size = 128
# # The n_layer of GPT-3.35B should be 16
# n_layer = 8
# sequence_len = 256
# learning_rate = 1e-3
# eval_iters = 20
# batch_size= 1
# epochs = 10

# GPT-2 medium
emb_size = 1024
head_size = 64
# The n_layer of GPT-3.35B should be 16
n_layer = 24
sequence_len = 256
learning_rate = 1e-4
eval_iters = 20
batch_size = 12
epochs = 10

def attention(query, key, value, dropout, mask=None):

    # query, key, value都有相同的形状
    B, T, C = query.shape
    # (B, T, C) @ (B, C, T) --> (B, T, T)
    scores = query @ key.transpose(-2, -1) / (C ** 0.5)
    if mask is not None:
        # 如果没有mask，则表示词元可以使用左右两边的背景，也就是双向注意力
        # 如果mask是上三角矩阵，则表示自回归模式的单向注意力
        # mask的形状是(T, T)
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
        # 这个上三角矩阵不参与模型训练
        self.register_buffer(
            'tril', torch.tril(torch.ones(sequence_len, sequence_len)))
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        
        B, T, C = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(x)    # (B, T, H)
        v = self.value(x)  # (B, T, H)
        mask = self.tril[:T, :T]
        out, _ = attention(q, k, v, self.dropout, mask)
        return out         # (B, T, H)

class MaskedMultiHeadAttention(nn.Module):

    def __init__(self, emb_size, head_size):
        
        super().__init__()
        # 确保特征长度是背景向量长度的倍数
        assert(emb_size % head_size == 0)
        # 定义单头注意力的个数
        n_head = emb_size // head_size
        heads = [MaskedAttention(emb_size, head_size) for _ in range(n_head)]
        self.heads = nn.ModuleList(heads)
        # 线性变换
        self.proj = nn.Linear(emb_size, emb_size)
        # 随机失活
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        
        # 将多个单头注意力的结果做张量拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C)
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
        # 层归一化
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
       
        # 残差连接
        x = x + self.mha(self.ln1(x))   # (B, T, C)
        out = x + self.ff(self.ln2(x))  # (B, T, C)
        return out
    
class CharGPT(nn.Module):

    def __init__(self, vs):

        super().__init__()
        # 文字嵌入层
        self.token_embedding = nn.Embedding(vs, emb_size)
        # 位置嵌入层
        self.position_embedding = nn.Embedding(sequence_len, emb_size)
        # 解码块
        blocks = [Block(emb_size, head_size) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*blocks)
        self.ln = nn.LayerNorm(emb_size)
        # 语言建模头
        self.lm_head = nn.Linear(emb_size, vs)

    def forward(self, x):
       
        B, T = x.shape
        # 定义词元的位置，形状为(T)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        # 词元语义特征
        tok_emb = self.token_embedding(x)       # (B, T,  C)
        # 位置特征
        pos_emb = self.position_embedding(pos)  # (   T,  C)
        x = tok_emb + pos_emb                   # (B, T,  C)
        x = self.blocks(x)                      # (B, T,  C)
        x = self.ln(x)                          # (B, T,  C)
        logits = self.lm_head(x)                # (B, T, vs)
        return logits

class char_tokenizer:

    def __init__(self, data):
        # 数据中出现的所有字符构成字典
        chars = sorted(list(set(''.join(data))))
        # 预留一个位置给结尾的特殊字符
        self.char2ind = {s : i + 1 for i, s in enumerate(chars)}
        self.char2ind['<|e|>'] = 0
        self.ind2char = {i : s for s, i in self.char2ind.items()}

    def encode(self, text):
        return [self.char2ind[c] for c in text]

    def decode(self, enc):
        if isinstance(enc, int):
            return self.ind2char[enc]
        return [self.ind2char[i] for i in enc]
        
# 数据集处理
# raw_datasets = load_dataset('code_search_net', 'python')
raw_datasets = load_dataset(
    path="/lihongliang/wangzc/GPT/dataset/CodeSearchNet/code_search_net.py",
    name="python",
    data_dir="/lihongliang/wangzc/GPT/dataset/CodeSearchNet"  # 会被传递给builder
)
datasets = raw_datasets['train'].filter(lambda x: 'apache/spark' in x['repository_name'])
tok = char_tokenizer(datasets['whole_func_string'])

# 初始化模型（添加DDP包装）
model = CharGPT(len(tok.char2ind))
model = model.to(rank)
model = DDP(model, device_ids=[rank])
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
# #----------------------------------------------------------------------------------------------------------------------------
# def save_model_structure(model, filename="./model_structure.txt"):
#     with open(filename, "w") as f:
#         f.write("{:<30} {:<20} {:<15}\n".format("Name", "Shape", "Type"))
#         f.write("="*60 + "\n")
        
#         for name, param in model.named_parameters():
#             f.write("{:<30} {:<20} {:<15}\n".format(
#                 name, str(tuple(param.size())), "Trainable Param"))
                
#         for name, buf in model.named_buffers():
#             f.write("{:<30} {:<20} {:<15}\n".format(
#                 name, str(tuple(buf.size())), "Buffer"))

# # 使用
# save_model_structure(model)
# #----------------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(Stage_list, idx, max_new_tokens=300):
    '''
    利用模型生成文本（反复使用模型进行预测）
    参数
    ----
    model  CharGPT 生成文本的模型
    idx  torch.LongTensor 当前字母在字典中的位置 形状为(1, T)
    max_new_tokens  nt 生成文本的最大长度
    返回
    ----
    out  list[int] 生成的文本
    '''
    # 将模型切换至评估模式
    for stage in Stage_list:
        stage.eval()
    for _ in range(max_new_tokens):
        # 限制背景长度，否则会报错
        logits = idx[:, -sequence_len:]
        # 在文本生成时，模型的计算效率很低，因为有很多重复计算
        for stage in Stage_list:
            # print(type(logits))
            logits = stage.forward(logits)
        # 只使用最后一个预测结果
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # 根据模型预测的概率，得到最终的预测结果（下一个字母）
        # 这一步运算有一定随机性
        ix = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, ix), dim=1)
        if ix.item() == 0:
            break
    # 将模型切换至训练模式
    for stage in Stage_list:
        stage.train()
    return idx.tolist()[0]

#使用模型来生成文本
# begin_text = torch.tensor(tok.encode('def'), device=device).unsqueeze(0)
# print(''.join(tok.decode(generate_batch(model, begin_text))))


def process(data, sequence_len=256):
    '''
    根据文本生成训练数据
    '''
    # text是字符串列表
    text = data['whole_func_string']
    inputs, labels = [], []
    for i in text:
        enc = tok.encode(i)
        # 0对应着文本结束
        enc += [0]
        # 将文本转换为多个训练数据
        for i in range(len(enc) - sequence_len):
            inputs.append(enc[i: i + sequence_len])
            # 预测标签是下一个字母，因此只需要挪动一个位置即可
            labels.append(enc[i + 1: i + 1 + sequence_len])
    return {'inputs': inputs, 'labels': labels}

# 将数据分为训练集和测试集
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
# 将文本转换为训练数据，里面包含inputs和labels
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
tokenized.set_format(type='torch')

print(tokenized['train']['inputs'].shape, tokenized['train']['labels'].shape)

# 构建数据读取器
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

# train_loader = DataLoader(tokenized['train'], batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(tokenized['test'], batch_size=batch_size, shuffle=True)
print("training step num is {}".format(len(train_loader)))
print("test step num is {}".format(len(test_loader)))
# # 获取一个批量的数据
# next(iter(test_loader))

# i = 0
# while i == 0:
#     i = 0

def estimate_loss(model):
    re = {}
    # 将模型切换至评估模式
    model.eval()
    re['train'] = _loss(model, train_loader)
    re['test'] = _loss(model, test_loader)
    # 将模型切换至训练模式
    model.train()
    return re

@torch.no_grad()
def _loss(model, data_loader):
    '''
    计算模型在不同数据集下面的评估指标
    '''
    loss = []
    data_iter= iter(data_loader)
    # 随机使用多个批量数据来预估模型效果
    for k in range(eval_iters):
        data = next(data_iter, None)
        if data is None:
            data_iter = iter(data_loader)
            data = next(data_iter, None)
        inputs, labels = data['inputs'].to(rank), data['labels'].to(rank)
        logits = model(inputs)
        # 根据cross_entropy的定义，需要对logits进行转置运算
        # 具体细节请参考cross_entropy的官方文档
        logits = logits.transpose(-2, -1)
        loss.append(F.cross_entropy(logits, labels).item())
    return torch.tensor(loss).mean().item()


#----------------------------------------------------------------------------------------------------------------------------
def save_model_structure(model, filename="./model_structure.txt"):
    with open(filename, "w") as f:
        f.write("{:<30} {:<20} {:<15}\n".format("Name", "Shape", "Type"))
        f.write("="*60 + "\n")
        
        for name, param in model.named_parameters():
            f.write("{:<30} {:<20} {:<15}\n".format(
                name, str(tuple(param.size())), "Trainable Param"))
                
        for name, buf in model.named_buffers():
            f.write("{:<30} {:<20} {:<15}\n".format(
                name, str(tuple(buf.size())), "Buffer"))
#----------------------------------------------------------------------------------------------------------------------------
def save_partial_model(model, path, num_layers_to_save):
    # 1. 保存模型参数（只保留前6层encoder）
    model_state = model.state_dict()
    partial_state = OrderedDict()
    
    for key in model_state:
        # 筛选属于前6层encoder的参数
        if 'module.blocks.' in key:
            layer_num = int(key.split('.')[2])  # 假设参数名格式为encoder.layers.0.xxx
            if layer_num < num_layers_to_save:
                partial_state[key] = model_state[key]
        # else:
        #     # 保留其他所有参数（如embedding、classifier等）
        #     partial_state[key] = model_state[key]
    
    # 3. 保存检查点
    torch.save({
        # 'epoch': epoch,
        'model_state': partial_state,
        # 'optimizer_state': opt_state,
    }, path)
    print(f"已保存前{num_layers_to_save}层检查点到 {path}")
#----------------------------------------------------------------------------------------------------------------------------

# print(estimate_loss(model))
import torch.distributed as dist
group = dist.new_group([0, 1, 2, 3])
# group = dist.new_group([0, 1])
# group = dist.new_group([0, 1, 2, 3, 4, 5, 6, 7])
global_rank = int(os.environ["RANK"])
def train_gpt(model, optimizer, data_loader, recovery, e, step, epochs=10):    
    # # 使用
    # if global_rank == 0: 
    #     save_model_structure(model)

    lossi = []
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # 重要：确保每个epoch有不同的shuffle
        if recovery and epoch < e:
            continue
        iter_start = time.time()
        for i, data in tqdm(enumerate(data_loader, 0)):
            if recovery and epoch == e and i <= step:
                continue

            # if i == 3 and epoch == 0 and global_rank == 0:
            #     snapshot_start = time.time()

            #     # 转移模型权重到CPU
            #     model_cpu = model.cpu()

            #     # 转移优化器状态到CPU
            #     for state in optimizer.state.values():
            #         for k, v in state.items():
            #             if isinstance(v, torch.Tensor):
            #                 state[k] = v.cpu()

            #     snapshot_end = time.time()
            #     print("snapshot all parameter's ckpt need {}s".format(snapshot_end - snapshot_start))
            #     return 0

            inputs, labels = data['inputs'].to(rank), data['labels'].to(rank)
            optimizer.zero_grad()
            # if i == 0:
            #     print("inputs shape is ", inputs.shape)

            forward_start = time.time()
            logits = model(inputs)
            # 根据cross_entropy的定义，需要对logits进行转置运算
            # 具体细节请参考cross_entropy的官方文档
            logits = logits.transpose(-2, -1)
            loss = F.cross_entropy(logits, labels)
            forward_end = time.time()
            print(f"forward time is {forward_end-forward_start}s")

            lossi.append(loss.item())

            backward_start = time.time()
            loss.backward()
            backward_end  = time.time()
            print(f"backward time is {backward_end-backward_start}s")

            allreduce_start = time.time()
            for p in model.parameters():
                p.grad = p.grad / world_size               
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group, async_op=False) 
            allreduce_end = time.time()
            # print('the allreduce cost of rank {} in a iter is {}s'.format(global_rank, allreduce_end - allreduce_start))

            optimizer.step()
            if i % 50 == 0:
                stats = estimate_loss(model)
                print(stats)

            # if i % 50 == 0 and i != 0:
            #     iter_end = time.time()
            #     print("50 iter training time is {}s".format(iter_end - iter_start))
            
            # # 开始计时
            # start_time = time.time()

            # # 转移模型参数
            # model = model.cpu()
            # # 转移优化器状态
            # for state in optimizer.state.values():
            #     for k, v in state.items():
            #         if torch.is_tensor(v):
            #             state[k] = v.cpu()        
            # # 计算耗时
            # transfer_time = time.time() - start_time
            # print(f"snapshot : {transfer_time}s")

# ########################
#             if i % 1 == 0 and global_rank == 0:
#                 lossf_type = 'origin'
#                 lossf_name = 'loss_' + lossf_type + '.npy'
#                 lossf_path = os.path.join("./loss", lossf_name)
#                 loss, epoch, iter = PreRecover_ck.save_loss(loss.item(), epoch, i // 1, lossf_path)
#             #----------------------------------------------------------------------------------------------------------------------------
#             if i % 1000 == 0 and i != 0 and global_rank == 0:
#                 ck_path = os.path.join("./checkpoint/gpt2_medium/1n1g", "epoch_{}_iter_{}_rank_{}.pth".format(epoch, i, 0))
#                 # ck_path = os.path.join("./checkpoint/gpt3_3.35B/1n1g", "epoch_{}_iter_{}_rank_{}.pth".format(epoch, i, 0))
#                 seeds = [random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()]
#                 ckpt_start = time.time()
#                 PreRecover_ck.save_checkpoint(model, 
#                                             optimizer, 
#                                             False, 
#                                             epoch, 
#                                             loss, 
#                                             i, 
#                                             seeds, 
#                                             train_loader,
#                                             1,
#                                             ck_path)
#                 ckpt_end = time.time()
#                 print("ckpt saving time is {}s".format(ckpt_end - ckpt_start))
            
#             if i % 20 == 0 and global_rank == 0:
#                 # sample_num = PreRecover_ck.save_layer(model.state_dict(), "layer.12.h.mlp.c_fc.weight", epoch, i)
#                 # sample_num = FRP_ck.cut_layer_to_cpu(model.state_dict(), "layer.12.h.mlp.c_fc.weight", _, i)
#                 partialCKPT_dir = "./part_ckpt/gpt2_medium/4n4g"
#                 num_layers_to_save = 6
#                 partialCKPT_path = os.path.join(partialCKPT_dir, "epoch_{}_iter_{}_block_{}.pth".format(epoch, i, num_layers_to_save))
#                 sample_num = save_partial_model(model, partialCKPT_path, num_layers_to_save)
#                 print("There are {} samples in {}".format(sample_num, "layer.12.h.mlp.c_fc.weight.npy"))
# ##########################
            if i % 1 == 0:
                print('epoch is {}, iter is {}, loss is {}'.format(epoch, i, loss))
            # if recovery:
            #     if i % 1 == 0:
            #         loss, epoch, iter = FRP_ck.save_loss(tr_loss, epoch, i // 1, 'predict')
            #         # loss, epoch, iter = FRP_ck.save_loss(tr_loss, _, i // 1, 'predict_mix')
            #         # loss, epoch, iter = FRP_ck.save_loss(tr_loss, _, i // 1, 'origin_recovery')
            #     if i % 10 == 0:
            #         print('epoch is {}, iter is {}, loss is {}'.format(epoch, iter, loss))
            #----------------------------------------------------------------------------------------------------------------------------
    return lossi
#----------------------------------------------------------------------------------------------------------------------------
'''
    作业恢复
'''
epoch = 0
i = 0 
# recovery = True
recovery = False
if recovery:
    ck_path = "/lihongliang/wangzc/GPT/dp/checkpoint/gpt2_medium/1n1g/epoch_2_iter_10000_rank_0.pth"
    ck = torch.load(ck_path)
    
    model.load_state_dict(ck['model_state_dict'])
    optimizer.load_state_dict(ck['optimizer_state_dict'])

    # 恢复随机数
    seeds = ck['seeds']
    random.setstate(seeds[0])
    np.random.set_state(seeds[1])
    # torch.random.set_rng_state(seeds[2])

    if seeds[2].device != torch.device('cpu'):
        #如果 RNG 状态不在 CPU -，移动到 CPU
        seeds[2] = seeds[2].cpu()
    torch.set_rng_state(seeds[2])
    torch.cuda.set_rng_state_all(seeds[3])

    epoch = ck['epoch']
    # loss = ck['loss']
    i = ck['iteration']

    del ck          # 删除引用
    torch.cuda.empty_cache()  # 清空PyTorch的CUDA缓存
    gc.collect()   # 触发Python垃圾回收
#----------------------------------------------------------------------------------------------------------------------------

l = train_gpt(model, optimizer, train_loader, recovery, epoch, i)

# begin_text = torch.tensor(tok.encode('def '), device=device).unsqueeze(0)
# print(''.join(tok.decode(generate_batch(model, begin_text))))