#!/bin/bash
# Node0 (1 GPUs)
if [ $HOSTNAME == "e57m63ikc1l0v-0" ]; then
    torchrun \
        --nnodes=8 \
        --node_rank=0 \
        --nproc_per_node=1 \
        --master_addr="172.22.153.25" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node1 (1 GPU)
elif [ $HOSTNAME == "4uhhj3iqhaa77-0" ]; then
    torchrun \
        --nnodes=8 \
        --node_rank=1 \
        --nproc_per_node=1 \
        --master_addr="172.22.153.25" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node2 (1 GPU)
elif [ $HOSTNAME == "801hqcpf0jac3-0" ]; then
    torchrun \
        --nnodes=8 \
        --node_rank=2 \
        --nproc_per_node=1 \
        --master_addr="172.22.153.25" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node3 (1 GPU)
elif [ $HOSTNAME == "2mtltbs1h3p0h-0" ]; then
    torchrun \
        --nnodes=8 \
        --node_rank=3 \
        --nproc_per_node=1 \
        --master_addr="172.22.153.25" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node4 (1 GPU)
elif [ $HOSTNAME == "clvc54n2bh49f-0" ]; then
    torchrun \
        --nnodes=8 \
        --node_rank=4 \
        --nproc_per_node=1 \
        --master_addr="172.22.153.25" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node5 (1 GPU)
elif [ $HOSTNAME == "22j4129qbpca3-0" ]; then
    torchrun \
        --nnodes=8 \
        --node_rank=5 \
        --nproc_per_node=1 \
        --master_addr="172.22.153.25" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node6 (1 GPU)
elif [ $HOSTNAME == "ep95o03oppm4q-0" ]; then
    torchrun \
        --nnodes=8 \
        --node_rank=6 \
        --nproc_per_node=1 \
        --master_addr="172.22.153.25" \
        --master_port=1231 \
        GPT2_1n1g.py

# Node7 (1 GPU)
elif [ $HOSTNAME == "fumqdf0r55ra2-0" ]; then
    torchrun \
        --nnodes=8 \
        --node_rank=7 \
        --nproc_per_node=1 \
        --master_addr="172.22.153.25" \
        --master_port=1231 \
        GPT2_1n1g.py
fi

