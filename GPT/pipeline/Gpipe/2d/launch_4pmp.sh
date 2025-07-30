#!/bin/bash
# Node0 (1 GPUs)
if [ $HOSTNAME == "ffh121i2k07hc-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=0 \
        --nproc_per_node=1 \
        --master_addr="172.21.133.25" \
        --master_port=1231 \
        Gpipe_2D.py

# Node1 (1 GPU)
elif [ $HOSTNAME == "4uhhj3iqhaa77-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=1 \
        --nproc_per_node=1 \
        --master_addr="172.21.133.25" \
        --master_port=1231 \
        Gpipe_2D.py

# Node2 (1 GPU)
elif [ $HOSTNAME == "2mtltbs1h3p0h-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=2 \
        --nproc_per_node=1 \
        --master_addr="172.21.133.25" \
        --master_port=1231 \
        Gpipe_2D.py

# Node3 (1 GPU)
elif [ $HOSTNAME == "22j4129qbpca3-0" ]; then
    torchrun \
        --nnodes=4 \
        --node_rank=3 \
        --nproc_per_node=1 \
        --master_addr="172.21.133.25" \
        --master_port=1231 \
        Gpipe_2D.py
fi

