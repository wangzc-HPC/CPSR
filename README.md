# COCI:Convergence-aware Optimal Checkpointing for Exploratory Deep Learning Training Jobs

This source code is available under the [MIT License](LICENSE.txt).

## Introduction

CPSR is a lightweight fault tolerance approach for distributed deep learning training. Built on PyTorch, CPSR introduces a prediction-based recovery mechanism that significantly reduces recomputation costs. The key innovation is using a trained predictor to estimate training states before failures, allowing self-correction during rehabilitation with minimal overhead (under 200MB GPU memory).

## Installation
### Prerequisites

* Python3.8.+
* PyTorch-1.10.+
* numpy-1.24.+
* scipy
* pandas
* csv
* tqdm
* scikit-learn
* pynvml
* datasets

###
  git clone https://github.com/wangzc-HPC/CPSR.git
  cd CPSR
  pip install -r requirements.txt

