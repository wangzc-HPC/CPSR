import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import datetime
import time
import random
import numpy as np
import gc

import sys
sys.path.append("/lihongliang/wangzc/")
from GPT.pipeline.Gpipe.GPT2 import *
from GPT import PreRecover_ck

# GPT-3.35B
emb_size = 4096
head_size = 128
# The n_layer of GPT-3.35B should be 16
n_layer = 16
sequence_len = 256
learning_rate = 1e-4
eval_iters = 20
batch_size = 24
epochs = 2

# # GPT-2 medium
# emb_size = 1024
# head_size = 64
# n_layer = 24
# sequence_len = 256
# learning_rate = 1e-4
# eval_iters = 20
# batch_size = 12
# epochs = 2

def init_model(model_config):
    # model=[]
    model = nn.ModuleList()
    for i,j in model_config.items():
        if i == "em_tokn":
            mdl = nn.Embedding(j[0], j[1])
        elif i == "em_pos":
            mdl = nn.Embedding(j[0], j[1])
        elif i == "ln":
            mdl = nn.LayerNorm(j[0])
        elif i == "lm_head":
            mdl = nn.Linear(j[0],j[1])
        elif (re.search("decoder",i)).group() == "decoder": # 可能会报错
            mdl = Block(j[0],j[1])
        model.append(mdl)
    return model

num_microbatches = 4  # 每个batch分成4个micro-batch

# 通信域创建
env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=300)) # gloo
global_rank = int(os.environ["RANK"])
print(f"global_rank is {global_rank}")
if global_rank == 0:
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

# if global_rank == 0:
#     DEVICE = 'cuda:0'
# elif global_rank == 1:
#     DEVICE = 'cuda:1'
# elif global_rank==2:
#     DEVICE = 'cuda:2'
# elif global_rank==3:
#     DEVICE = 'cuda:3'

# 替换原有的DEVICE分配逻
def get_device():
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        return f'cuda:{local_rank}'
    return 'cpu'

DEVICE = get_device()

print("The current rank is {}".format(DEVICE))

#===================== 2D start =====================
# 在初始化分布式环境后添加
world_size = dist.get_world_size()
model_parallel_size = 4  # 每个节点4个GPU做模型并行
data_parallel_size = world_size // model_parallel_size

# 创建模型并行组（节点内通信）
model_parallel_groups = [
    dist.new_group(list(range(i*model_parallel_size, (i+1)*model_parallel_size))) 
    for i in range(data_parallel_size)
]

# 创建数据并行组（跨节点通信）
data_parallel_groups = [
    dist.new_group(list(range(i, world_size, model_parallel_size)))
    for i in range(model_parallel_size)
]

# 为当前rank分配组
model_parallel_group = model_parallel_groups[global_rank // model_parallel_size]
data_parallel_group = data_parallel_groups[global_rank % model_parallel_size]
#===================== 2D end =====================

class Stage:
    def __init__(self, ID, model, model_idx, learning_rate, device, batch_size):
        self.stage_ID = ID
        self.device = device
        self.model_idx = model_idx
        self.is_training = True
        self.micro_batch_size = batch_size // num_microbatches

        #===================== 2D start =====================
        # 添加通信组
        self.model_parallel_group = model_parallel_group
        self.data_parallel_group = data_parallel_group
        self.local_rank = global_rank % model_parallel_size  # 节点内局部rank
        #===================== 2D end =====================

        if self.stage_ID == 1:
            # 文字嵌入层
            self.token_embedding = nn.Embedding(vs, emb_size)
            # 位置嵌入层
            self.position_embedding = nn.Embedding(sequence_len, emb_size)
            self.sub_model = nn.Sequential(*[model[i] for i in range(2,len(self.model_idx))])
        else:
            self.sub_model = nn.Sequential(*[model[i] for i in model_idx])
        # self.sub_model_save = nn.Sequential(*[model[i] for i in model_idx])

        # 优化器现在只需要一个，因为模型已经被Sequential包装
        if self.stage_ID == 1:
            self.sub_model_opt = nn.Sequential(*[model[i] for i in model_idx])
            self.optimizer = optim.Adam(self.sub_model_opt.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.sub_model.parameters(), lr=learning_rate)
        # self.optimizer_list= [optim.Adam(model[i].parameters(), lr=learning_rate) for i in model_idx]

        self.out_x = torch.zeros(self.micro_batch_size, sequence_len, emb_size).to(device)
        self.grad_y = torch.zeros(self.micro_batch_size, sequence_len, emb_size).to(device)
        
        self.lossi = []    
        self.fwd_cache = []  # Cache forward propagation results for backpropagation.

    def to(self,device):
        if self.stage_ID == 1:
            self.token_embedding.to(device)
            self.position_embedding.to(device)
        self.sub_model.to(device)
    
    def eval(self):
        self.sub_model.eval()
    
    def train(self):
        self.sub_model.train()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def forward(self, x):
        if self.stage_ID == 1:
            B, T = x.shape
            # 定义词元的位置，形状为(T)
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)

            # 词元语义特征
            tok_emb = self.token_embedding(x)       # (B, T,  C)
            # 位置特征
            pos_emb = self.position_embedding(pos)  # (   T,  C)

            x = tok_emb + pos_emb
            x = self.sub_model(x)
            # print("输出tensor的形状:{}".format(x.shape))
        else:
            x = self.sub_model(x)
        
        x.retain_grad() 
        self.fwd_cache.append(x)  # 缓存结果
        return x
        
    def forward_send(self, out):
        # dist.send(tensor=out, dst=self.stage_ID, tag=self.stage_ID)
        """只在模型并行组内发送数据"""
        # dst = (global_rank - self.local_rank) + ((self.local_rank + 1) % model_parallel_size) #global_rank+1
        dst = global_rank + 1
        dist.send(tensor=out, dst=dst, tag=self.stage_ID, group=self.model_parallel_group)


    def forward_recv(self):
        # dist.recv(tensor=self.out_x, src=self.stage_ID-2, tag=self.stage_ID-1)
        # self.out_x.to(self.device)
        """只在模型并行组内接收数据"""
        # src = (global_rank - self.local_rank) + ((self.local_rank - 1) % model_parallel_size) #global_rank-1
        src = global_rank - 1
        dist.recv(tensor=self.out_x, src=src, tag=self.stage_ID-1, group=self.model_parallel_group)
        self.out_x = self.out_x.to(self.device)

    def backward(self, microbatch_idx=None):
        cached_output = self.fwd_cache[microbatch_idx]
        cached_output.backward(self.grad_y)
            
    def backward_send(self, microbatch_idx):
        # dist.send(tensor=self.fwd_cache[microbatch_idx].grad, dst=self.stage_ID-2, tag=self.stage_ID)
        grad = self.fwd_cache[microbatch_idx].grad
        # dst = (global_rank - self.local_rank) + ((self.local_rank - 1) % model_parallel_size) #global_rank-1
        dst = global_rank - 1
        dist.send(grad, dst=dst, group=self.model_parallel_group)

    def backward_recv(self):
        # dist.recv(tensor=self.grad_y, src=self.stage_ID, tag=self.stage_ID+1)
        # self.grad_y.to(self.device)
        # src = (global_rank - self.local_rank) + ((self.local_rank + 1) % model_parallel_size) #global_rank+1
        src = global_rank + 1
        dist.recv(self.grad_y, src=src, group=self.model_parallel_group)
        self.grad_y = self.grad_y.to(self.device)

    def all_reduce_gradients(self):
        """在数据并行组内同步梯度"""
        for param in self.sub_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
                param.grad /= data_parallel_size

    # def all_reduce_gradients(self):
    #     """只在数据并行组内同步梯度"""
    #     if dist.get_world_size(self.data_parallel_group) > 1:  # 仅在需要时同步
    #         for param in self.sub_model.parameters():
    #             if param.grad is not None:
    #                 dist.all_reduce(
    #                     param.grad, 
    #                     op=dist.ReduceOp.SUM,
    #                     group=self.data_parallel_group
    #                 )
    #                 param.grad /= dist.get_world_size(self.data_parallel_group)


# 将数据分为训练集和测试集
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
# 将文本转换为训练数据，里面包含inputs和labels
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
tokenized.set_format(type='torch', device=DEVICE)
if global_rank == 0: 
    print("Train dataset inputs and labels shape are {} and {}".format(tokenized['train']['inputs'].shape, tokenized['train']['labels'].shape))
# 构建数据读取器

train_sampler = DistributedSampler(
    dataset=tokenized['train'],
    num_replicas=data_parallel_size,  # 数据并行组大小
    rank=global_rank // model_parallel_size,  # 数据并行组内的rank
    shuffle=True
) if data_parallel_size > 1 else None

train_loader = DataLoader(
    dataset=tokenized['train'],
    batch_size=batch_size,
    sampler=train_sampler, 
    num_workers=0
)

test_sampler = DistributedSampler(
    tokenized['test'],
    num_replicas=data_parallel_size,  # 必须与train_sampler相同
    rank=global_rank // model_parallel_size,
    shuffle=False  # 测试集通常不shuffle！
) if data_parallel_size > 1 else None

test_loader = DataLoader(
    tokenized['test'],
    batch_size=batch_size,
    sampler=test_sampler,
    num_workers=0
)

if global_rank == 0: 
    print("training step num is {}".format(len(train_loader)))
    print("test step num is {}".format(len(test_loader)))
# # 获取一个批量的数据
# next(iter(test_loader))

'''
    emb_size = 4096
    head_size = 128
    n_layer = 16
    sequence_len = 256
    learning_rate = 1e-3
    eval_iters = 20
    batch_size= 12
'''
vs = len(tok.char2ind)
# GPT-3.35B
'''
    model_config={"em_tokn":[vs,emb_size],"em_pos":[sequence_len,emb_size],"decoder1":[emb_size,head_size],"decoder2":[emb_size,head_size],"decoder3":[emb_size,head_size],
              "decoder4":[emb_size,head_size],"decoder5":[emb_size,head_size],"decoder6":[emb_size,head_size],"decoder7":[emb_size,head_size],"decoder8":[emb_size,head_size],
              "decoder9":[emb_size,head_size],"decoder10":[emb_size,head_size],"decoder11":[emb_size,head_size],"decoder12":[emb_size,head_size],"decoder13":[emb_size,head_size],
              "decoder14":[emb_size,head_size],"decoder15":[emb_size,head_size],"decoder16":[emb_size,head_size],"ln":[emb_size],"lm_head":[emb_size,vs]}
'''

# GPT-2 medium
model_config = {
    "em_tokn": [vs, emb_size],
    "em_pos": [sequence_len, emb_size],
    **{f"decoder{i}": [emb_size, head_size] for i in range(1, n_layer+1)},
    "ln": [emb_size],
    "lm_head": [emb_size, vs]
}

gpt = init_model(model_config=model_config)
# # GPT2-MEDIUM 4stage 2dp
# model_idx1=[0,1,2,3,4,5,6,7]
# model_idx2=[8,9,10,11,12,13]
# model_idx3=[14,15,16,17,18,19]
# model_idx4=[20,21,22,23,24,25,26,27]

# GPT3-3.35B 4stage 2dp
model_idx1=[0,1,2,3,4,5]
model_idx2=[6,7,8,9]
model_idx3=[10,11,12,13]
model_idx4=[14,15,16,17,18,19]

# # GPT3-3.35B 3stage 1dp
# model_idx1=[0,1,2,3,4,5,6]
# model_idx2=[7,8,9,10,11,12]
# model_idx3=[13,14,15,16,17,18,19]

# model_idx1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
# model_idx2 = [14,15,16,17,18,19,20,21,22,23,24,25,26,27]

rank_to_stage = {
    0: 1,  # rank0 -> stage1
    1: 2,  # rank1 -> stage2
    2: 3,  # rank2 -> stage3
    3: 4,  # rank3 -> stage4
    4: 1,  # rank0 -> stage1
    5: 2,  # rank1 -> stage2
    6: 3,  # rank2 -> stage3
    7: 4,  # rank3 -> stage4
}

# 获取当前rank对应的stage
current_stage = rank_to_stage[global_rank]
print("current rank is {}, current stage is {}".format(global_rank, current_stage))

Stage_list = []
for rank in range(world_size):
    if rank_to_stage[rank] == 1:
        Stage_list.append(Stage(1, gpt, model_idx1, learning_rate, DEVICE, batch_size))
    if rank_to_stage[rank] == 2:
        Stage_list.append(Stage(2, gpt, model_idx2, learning_rate, DEVICE, batch_size))
    if rank_to_stage[rank] == 3:
        Stage_list.append(Stage(3, gpt, model_idx2, learning_rate, DEVICE, batch_size))
    if rank_to_stage[rank] == 4:
        Stage_list.append(Stage(4, gpt, model_idx2, learning_rate, DEVICE, batch_size))

# 每个rank只初始化自己的stage
my_stage = Stage_list[global_rank]

for i in range(len(Stage_list)):
    if i == global_rank:
        Stage_list[i].to(DEVICE)

#----------------------------------------------------------------------------------------------------------------------------
'''
    作业恢复
'''

def load_state_dict_chunked(model, state_dict, chunk_size=1000000):
    """分块加载参数，避免内存峰值"""
    model_state = model.state_dict()
    keys = list(state_dict.keys())
    
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i:i + chunk_size]
        chunk = {k: state_dict[k] for k in chunk_keys}
        
        # 加载当前分块
        model.load_state_dict(chunk, strict=False)
        
        # 手动清理临时变量
        del chunk
        torch.cuda.empty_cache()  # 清理GPU缓存
        gc.collect()


def load_optimizer_chunked(optimizer, optimizer_state, chunk_size=1000):
    """分批次加载优化器状态"""
    # 先加载参数组（非张量，内存占用小）
    optimizer.load_state_dict({
        'param_groups': optimizer_state['param_groups']
    })
    
    # 分批次加载状态张量
    state_keys = list(optimizer_state['state'].keys())
    for i in range(0, len(state_keys), chunk_size):
        chunk_keys = state_keys[i:i + chunk_size]
        chunk_state = {
            k: {name: t.to(DEVICE) if isinstance(t, torch.Tensor) else t
                for name, t in optimizer_state['state'][k].items()}
            for k in chunk_keys
        }
        
        # 更新当前分块状态
        optimizer.state.update(chunk_state)
        
        # 清理临时内存
        del chunk_state
        torch.cuda.empty_cache()


epoch = 0
i = 0 
# recovery = True
recovery = False
if recovery:
    ck_dir = "/lihongliang/wangzc/GPT/pipeline/Gpipe/2d/checkpoint/gp3_3.35B/3n3g"
    ck_name = 'epoch_0_iter_10_rank_' + (str)(global_rank) + '.pth'
    ck_path = os.path.join(ck_dir, ck_name)
    # ck_path = "/lihongliang/wangzc/GPT/dp/checkpoint/gpt2_medium/1n1g/epoch_2_iter_10000_rank_0.pth"

    with open(ck_path, 'rb') as f:
        ck = torch.load(f, map_location='cpu')

    # 1. 分块加载模型参数
    if global_rank == 0:
        load_state_dict_chunked(my_stage.token_embedding, ck['token_embedding'])
        load_state_dict_chunked(my_stage.position_embedding, ck['position_embedding'])
        load_state_dict_chunked(my_stage.sub_model, ck['blocks_state'])
        
        # 移动到GPU
        my_stage.token_embedding.to(DEVICE)
        my_stage.position_embedding.to(DEVICE)
        my_stage.sub_model.to(DEVICE)
    else:
        load_state_dict_chunked(my_stage.sub_model, ck['blocks_state'])
        my_stage.sub_model.to(DEVICE)

    # 2. 分批次加载优化器状态
    load_optimizer_chunked(my_stage.optimizer, ck['optimizer_state_dict'])
    # my_stage.optimizer.load_state_dict(ck['optimizer_state_dict'])
     # 分步加载优化器状态
    # optimizer_state = ck['optimizer_state_dict']
    # for k, v in optimizer_state['state'].items():
    #     for name, tensor in v.items():
    #         if isinstance(tensor, torch.Tensor):
    #             optimizer_state['state'][k][name] = tensor.to(DEVICE)  # 逐个张量转移
    #             # 强制释放CPU上的原始张量
    #             del tensor  # 删除临时引用
    #             torch.cuda.empty_cache()  # 清空未使用的GPU缓存（可选）
    #             gc.collect()

    # my_stage.optimizer.load_state_dict(optimizer_state)
    # del optimizer_state  # 删除大对象

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

    epoch_r = ck['epoch']
    # loss = ck['loss']
    i_r = ck['iteration']

    del ck          # 删除引用
    torch.cuda.empty_cache()  # 清空PyTorch的CUDA缓存
    gc.collect()   # 触发Python垃圾回收
#----------------------------------------------------------------------------------------------------------------------------
loss_list = []

ck_interval = 1000
loss = 0
for epoch in range(epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    if recovery and epoch < epoch_r:
        continue

    iter_start = time.time()
    for i, data in tqdm(enumerate(train_loader, 0)):
        if recovery and epoch == epoch_r and i <= i_r:
            continue
        train_start = time.time()
        inputs, labels = data['inputs'].to(DEVICE), data['labels'].to(DEVICE)
        micro_inputs = torch.chunk(inputs, num_microbatches)
        micro_labels = torch.chunk(labels, num_microbatches)
        
        # === Forward ===
        for mb_idx in range(num_microbatches):
            # First stage
            if current_stage == 1:  
                f_start = time.time()
                out = my_stage.forward(micro_inputs[mb_idx])
                f_end = time.time()
                print(f"forward time is {f_end-f_start}s")
                my_stage.forward_send(out)
            
            # Last stage
            elif current_stage == model_parallel_size:
                my_stage.forward_recv()
                out = my_stage.forward(my_stage.out_x)
                
            # Middle stage   
            else:  
                my_stage.forward_recv()
                out = my_stage.forward(my_stage.out_x)
                my_stage.forward_send(out)

        # === Backward ===
        for mb_idx in reversed(range(num_microbatches)):
            # Last stage
            if current_stage == model_parallel_size:  
                logits = my_stage.fwd_cache[mb_idx].transpose(-2, -1)
                loss = F.cross_entropy(logits, micro_labels[mb_idx])

                if global_rank == world_size - 1:
                    if i % 1 == 0 and mb_idx == num_microbatches - 1: #1epoch 20285 loss
                        lossf_path = "/lihongliang/wangzc/GPT/pipeline/Gpipe/2d/loss/GPT3-3.35B-4.npy"
                        PreRecover_ck.save_loss(loss.item(), epoch, i // 1, lossf_path)

                loss.backward()
                my_stage.backward_send(mb_idx)

                if i % 100 == 0 and global_rank == world_size - 1:
                    print(f"epoch: {epoch}, iter: {i}, loss: {loss.item()}")
            
            # First stage
            elif current_stage == 1:  
                my_stage.backward_recv()
                b_start = time.time()
                my_stage.backward(mb_idx)
                b_end = time.time()
                print(f"backward time is {b_end-b_start}s")

            # Middle stage  
            else:  
                my_stage.backward_recv()
                my_stage.backward(mb_idx)
                my_stage.backward_send(mb_idx)   

        # === Update ===
        if my_stage.is_training:
            if data_parallel_size > 1:
                comm_start = time.time()
                my_stage.all_reduce_gradients()
                comm_end = time.time()
                print("allreduce cost is {}s".format(comm_end - comm_start))

            my_stage.optimizer.step()
            my_stage.optimizer.zero_grad()
            my_stage.fwd_cache.clear()
        train_end = time.time()
        # print("train cost is {}s".format(train_end - train_start))
        

        if i % 50 == 0 and i != 0:
            iter_end = time.time()
            print("50 iter end time is {}s".format(iter_end - iter_start))
        
        # === Saving ===
        if i % ck_interval == 0 and i != 0 and global_rank < model_parallel_size:
            # ck_path = os.path.join("./checkpoint/gpt2_medium/8n8g", "epoch_{}_iter_{}_rank_{}.pth".format(epoch, i, global_rank))
            ck_path = os.path.join("./checkpoint/gp3_3.35B/4n4g", "epoch_{}_iter_{}_rank_{}.pth".format(epoch, i, global_rank))
            # ck_path = os.path.join("./checkpoint/gp3_3.35B/3n3g", "epoch_{}_iter_{}_rank_{}.pth".format(epoch, i, global_rank))
            seeds = [random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()]

            ckpt_start = time.time()
            if global_rank == 0:
                checkpoint = {
                    'token_embedding': my_stage.token_embedding.state_dict(),  # 单独保存嵌入层
                    'position_embedding': my_stage.position_embedding.state_dict(),  # 单独保存嵌入层
                    'blocks_state': my_stage.sub_model.state_dict(),  # 保存主体块
                    'optimizer_state_dict': my_stage.optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),  # 如果有调度器
                    'epoch': epoch,
                    'iteration' : i,
                    'loss': loss,  # 当前损失
                    'seeds': seeds,
                }

                torch.save(checkpoint, ck_path)
                print(f'Checkpoint saved to {ck_path}')
            else:
                checkpoint = {
                    'blocks_state': my_stage.sub_model.state_dict(),  # 保存主体块
                    'optimizer_state_dict': my_stage.optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),  # 如果有调度器
                    'epoch': epoch,
                    'iteration' : i,
                    'loss': loss,  # 当前损失
                    'seeds': seeds,
                }

                torch.save(checkpoint, ck_path)
                print(f'Checkpoint saved to {ck_path}')
            # PreRecover_ck.save_checkpoint(my_stage.sub_model_save, 
            #                               my_stage.optimizer, 
            #                               False, 
            #                               epoch, 
            #                               loss, 
            #                               i, 
            #                               seeds, 
            #                               train_loader,
            #                               1,
            #                               ck_path)
            ckpt_end = time.time()
            print("rank_{} ckpt saving time is {}s".format(global_rank, ckpt_end - ckpt_start))

