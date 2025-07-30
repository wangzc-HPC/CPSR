import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import os
import torch.distributed as dist
import datetime
import time
import numpy as np
import random

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
learning_rate = 1e-3
eval_iters = 20
batch_size = 2
epochs = 2

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

num_microbatches = 2  # 每个batch分成4个micro-batch

class Stage:
    def __init__(self, ID, model, model_idx, learning_rate, device, batch_size):
        self.stage_ID = ID
        self.device = device
        self.model_idx = model_idx
        self.is_training = True
        self.micro_batch_size = batch_size // num_microbatches

        if self.stage_ID == 1:
            # 文字嵌入层
            self.token_embedding = nn.Embedding(vs, emb_size)
            # 位置嵌入层
            self.position_embedding = nn.Embedding(sequence_len, emb_size)
            self.sub_model = nn.Sequential(*[model[i] for i in range(2,len(self.model_idx))])
        else:
            self.sub_model = nn.Sequential(*[model[i] for i in model_idx])

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
        dist.send(tensor=out, dst=self.stage_ID, tag=self.stage_ID)

    def forward_recv(self):
        dist.recv(tensor=self.out_x, src=self.stage_ID-2, tag=self.stage_ID-1)
        self.out_x.to(self.device)

    def backward(self, microbatch_idx=None):
        cached_output = self.fwd_cache[microbatch_idx]
        cached_output.backward(self.grad_y)
            
    def backward_send(self, microbatch_idx):
        dist.send(tensor=self.fwd_cache[microbatch_idx].grad, dst=self.stage_ID-2, tag=self.stage_ID)

    def backward_recv(self):
        dist.recv(tensor=self.grad_y, src=self.stage_ID, tag=self.stage_ID+1)
        self.grad_y.to(self.device)


# 通信域创建
env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=30)) # gloo
global_rank = int(os.environ["RANK"])
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


# 将数据分为训练集和测试集
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
# 将文本转换为训练数据，里面包含inputs和labels
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
tokenized.set_format(type='torch', device=DEVICE)
if global_rank == 0: 
    print("Train dataset inputs and labels shape are {} and {}".format(tokenized['train']['inputs'].shape, tokenized['train']['labels'].shape))
# 构建数据读取器
train_loader = DataLoader(tokenized['train'], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tokenized['test'], batch_size=batch_size, shuffle=True)
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
# model_idx1=[0,1,2,3,4,5]
# model_idx2=[6,7,8,9]
# model_idx3=[10,11,12,13]
# model_idx4=[14,15,16,17,18,19]
model_idx1 = [0,1,2,3,4,5,6,7,8,9,10]
model_idx2 = [11,12,13,14,15,16,17,18,19]
s1 = Stage(1, gpt, model_idx1, learning_rate, DEVICE, batch_size)
s2 = Stage(2, gpt, model_idx2, learning_rate, DEVICE, batch_size)
# s3 = Stage(3,gpt,model_idx3,learning_rate,DEVICE,batch_size)
# s4 = Stage(4,gpt,model_idx4,learning_rate,DEVICE,batch_size)

# Stage_list = [s1,s2,s3,s4]
Stage_list = [s1,s2]

for i in range(len(Stage_list)):
    if i == global_rank:
        Stage_list[i].to(DEVICE)

ck_interval = 1000
loss_list = []
train_start = time.time()
for epoch in range(epochs):
    for i, data in tqdm(enumerate(train_loader, 0)):
        inputs, labels = data['inputs'], data['labels']
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        micro_inputs = torch.chunk(inputs, num_microbatches)
        micro_labels = torch.chunk(labels, num_microbatches)

        # === Forward ===
        for mb_idx in range(num_microbatches):
            # First stage
            if global_rank == 0:  
                out = Stage_list[0].forward(micro_inputs[mb_idx])
                Stage_list[0].forward_send(out)
            
            # Last stage
            elif global_rank == len(Stage_list) - 1:  
                Stage_list[-1].forward_recv()
                out = Stage_list[-1].forward(Stage_list[-1].out_x)
                
            # Middle stage   
            else:  
                Stage_list[global_rank].forward_recv()
                out = Stage_list[global_rank].forward(Stage_list[global_rank].out_x)
                Stage_list[global_rank].forward_send(out)

        # === Backward ===
        for mb_idx in reversed(range(num_microbatches)):
            # Last stage
            if global_rank == len(Stage_list) - 1:  
                logits = Stage_list[-1].fwd_cache[mb_idx].transpose(-2, -1)
                loss = F.cross_entropy(logits, micro_labels[mb_idx])
                loss.backward()
                Stage_list[-1].backward_send(mb_idx)

                
                loss_list.append(loss.item())
                if i % 1 == 0:
                    loss_np = np.array(loss_list)
                    # np.save("./loss/GPT2-medium-8.npy", loss_np)
                    np.save("./loss/GPT3-3.35B-2.npy", loss_np)
            
            # First stage
            elif global_rank == 0:  
                Stage_list[0].backward_recv()
                Stage_list[0].backward(mb_idx)

            # Middle stage  
            else:  
                Stage_list[global_rank].backward_recv()
                Stage_list[global_rank].backward(mb_idx)
                Stage_list[global_rank].backward_send(mb_idx)     

        # === Update ===
        if Stage_list[global_rank].is_training:
            Stage_list[global_rank].optimizer.step()
            Stage_list[global_rank].optimizer.zero_grad()
            Stage_list[global_rank].fwd_cache.clear()
        
        if i % 50 == 0 and i != 0:
            end_50iter = time.time()
            print("50 iter training time is {}s".format(end_50iter - train_start))

        if i % ck_interval == 0 and epoch != 0:
            ck_path = os.path.join("./checkpoint/gpt2_medium/8n8g", "epoch_{}_iter_{}_rank_{}.pth".format(epoch, i, global_rank))
            seeds = [random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()]

            ckpt_start = time.time()
            PreRecover_ck.save_checkpoint(Stage_list[global_rank].sub_model_save, 
                                        Stage_list[global_rank].optimizer, 
                                        False, 
                                        epoch, 
                                        loss, 
                                        i, 
                                        seeds, 
                                        train_loader,
                                        1,
                                        ck_path)
            ckpt_end = time.time()
            print("rank_{} ckpt saving time is {}s".format(global_rank, ckpt_end - ckpt_start))
