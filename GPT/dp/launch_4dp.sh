#!/bin/bash
# Node0 (1 GPUs) a30_1
if [ $HOSTNAME == "4uhhj3iqhaa77-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=0 \
        --nproc_per_node=1 \
        --master_addr="172.23.36.27" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node1 (1 GPU) a30_3
elif [ $HOSTNAME == "2mtltbs1h3p0h-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=1 \
        --nproc_per_node=1 \
        --master_addr="172.23.36.27" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node2 (1 GPU) a30_1
elif [ $HOSTNAME == "22j4129qbpca3-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=2 \
        --nproc_per_node=1 \
        --master_addr="172.23.36.27" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node3 (1 GPU) a30_0
elif [ $HOSTNAME == "ffh121i2k07hc-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=3 \
        --nproc_per_node=1 \
        --master_addr="172.23.36.27" \
        --master_port=1231 \
        GPT2_1n1g.py
fi

