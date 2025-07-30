#!/bin/bash
# Node0 (1 GPUs)
if [ $HOSTNAME == "e57m63ikc1l0v-0" ]; then
    torchrun \
        --nnodes=2 \
        --node_rank=0 \
        --nproc_per_node=1 \
        --master_addr="172.21.133.25" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node1 (1 GPU)
elif [ $HOSTNAME == "4uhhj3iqhaa77-0" ]; then
    torchrun \
        --nnodes=2 \
        --node_rank=1 \
        --nproc_per_node=1 \
        --master_addr="172.21.133.25" \
        --master_port=1231 \
        GPT2_1n1g.py
fi

