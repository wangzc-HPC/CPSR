#!/bin/bash
# Node0 (1 GPUs) 0
if [ $HOSTNAME == "e57m63ikc1l0v-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=0 \
        --nproc_per_node=1 \
        --master_addr="172.21.133.25" \
        --master_port=1111 \
        Gpipe.py

# Node1 (1 GPU) 4
elif [ $HOSTNAME == "clvc54n2bh49f-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=1 \
        --nproc_per_node=1 \
        --master_addr="172.21.133.25" \
        --master_port=1111 \
        Gpipe.py

# Node2 (1 GPU) 3
elif [ $HOSTNAME == "801hqcpf0jac3-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=2 \
        --nproc_per_node=1 \
        --master_addr="172.21.133.25" \
        --master_port=1111 \
        Gpipe.py

# Node3 (1 GPU) 5
elif [ $HOSTNAME == "2mtltbs1h3p0h-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=3 \
        --nproc_per_node=1 \
        --master_addr="172.21.133.25" \
        --master_port=1111 \
        Gpipe.py
fi

