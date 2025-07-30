#!/bin/bash
# Node0 (1 GPUs)
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --master_addr="172.23.36.28" \
    --master_port=1231 \
    GPT2_1n1g.py
    


