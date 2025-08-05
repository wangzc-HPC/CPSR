import numpy as np
import pandas as pd
import torch
import os
import csv
import gc
import re
import random
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

global_ck_directory = "/CPSR/GPT/ck/"


def save_checkpoint(model, optimizer, scheduler, epoch, loss, iter, seeds, train_loader, grad_accumulation_step,
                    filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'iteration': iter,
        'loss': loss,  # current loss
        'seeds': seeds,
    }
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved to {filename}')


def cut_layer(model_state_dict, layer_name):
    t = model_state_dict[layer_name].cpu()
    t_flat = torch.flatten(t)
    t_np = t_flat.numpy()
    t_np = t_np.reshape(1, t_np.size)

    if t_np.size == t.numel():
        print(f"sample size is {t.numel()}")
    else:
        print("ERROR!")

    ckf_name = f"{layer_name}.npy"
    ck_directory = "./ck/1"
    if not os.path.exists(ck_directory):
        os.makedirs(ck_directory)
    ckf_path = os.path.join("./ck/1", ckf_name)
    sample_num = 0
    if os.path.exists(ckf_path):
        ck_history = np.load(ckf_path)
        t_combine = np.concatenate((ck_history, t_np))
        sample_num = t_combine.shape[0]
        print(f"t_combine.shape is {t_combine.shape}")
    else:
        t_combine = t_np
        sample_num = 1
    np.save(ckf_path, t_combine)
    return sample_num


def cut_layer_to_cpu(model_state_dict, layer_name, epoch, iter):
    t = model_state_dict[layer_name].cpu()
    t_flat = torch.flatten(t)
    t_np = t_flat.numpy()
    t_np = t_np.reshape(1, t_np.size)

    if t_np.size == t.numel():
        print(f"sample size is {t.numel()}")
    else:
        print("ERROR!")

    ckf_name = f"{epoch}epoch_{iter}iteration.npy"
    ckf_path = os.path.join("./ck", layer_name, ckf_name)
    ck_directory = os.path.dirname(ckf_path)
    if not os.path.exists(ck_directory):
        os.makedirs(ck_directory)

    np.save(ckf_path, t_np)
    return True


def save_layer(model_state_dict, layer_name, epoch, iter):
    t = model_state_dict[layer_name].cpu()
    t_np = t.numpy()

    if t_np.size == t.numel():
        print(f"sample size is {t.numel()}")
    else:
        print("ERROR!")

    ckf_name = f"{iter}_iteration.npy"
    ckf_path = os.path.join("./ck", layer_name, str(epoch) + "_epoch", ckf_name)
    ck_directory = os.path.dirname(ckf_path)
    if not os.path.exists(ck_directory):
        os.makedirs(ck_directory)

    np.save(ckf_path, t_np)
    return True


def __normalize(rawdata):
    min = rawdata.min()
    max = rawdata.max()
    norm_data = (rawdata - min) / (max - min)
    return norm_data, min, max


def min_max_inverse_normalize(normalized_data, original_min, original_max):
    return normalized_data * (original_max - original_min) + original_min


def assemble_onefile(cks_directory):
    # Read all checkpoints in training process
    if not os.path.exists(cks_directory):
        print("The checkpoints directory does not exist.")
        return False

    cks = []
    ck_files = os.listdir(cks_directory)
    file_num = 0
    with tqdm(total=len(ck_files), desc="Assembling") as pbar:
        for ck in ck_files:
            ckf_path = os.path.join(cks_directory, ck)
            ck = np.load(ckf_path)
            if file_num == 0:
                cks = ck
            if cks.ndim != ck.ndim:
                print("Error in array dimensions!")
                break
            if file_num != 0:
                cks = np.concatenate((cks, ck))
            file_num += 1
            pbar.update(1)

    model_name = cks_directory.split('/')[-1]
    assemble_directory = "/CPSR/GPT/ck/assemble/"
    assemble_path = os.path.join(assemble_directory, model_name + ".npy")
    np.save(assemble_path, cks)

    return cks


def assemble(cks_directory, iteration):
    # Read all checkpoints in training process
    if not os.path.exists(cks_directory):
        print("The checkpoints directory does not exist.")
        return False

    model_name = cks_directory.split('/')[-1]
    assemble_directory = os.path.join(global_ck_directory, "assemble/")

    cks_directorys = os.listdir(cks_directory)
    for epoch in cks_directorys:
        epoch_num = epoch.split('_')[0]
        save_path = os.path.join(assemble_directory, model_name, epoch_num)
        os.makedirs(save_path, exist_ok=True)
        assemble_path = os.path.join(save_path, model_name + ".npy")
        normalization_path = os.path.join(save_path, "normalization" + ".npy")

        epoch_cks_dir = os.path.join(cks_directory, epoch)
        print(epoch_cks_dir)
        ck_files = os.listdir(epoch_cks_dir)
        # Sort files by modification time
        sorted_files_by_time = sorted(ck_files, key=lambda x: os.path.getmtime(os.path.join(epoch_cks_dir, x)))

        cks = []
        file_num = 0
        with tqdm(total=len(sorted_files_by_time), desc="Assembling") as pbar:
            for ck in sorted_files_by_time:
                iter_num = int(ck.split('_')[0])

                ckf_path = os.path.join(epoch_cks_dir, ck)
                ck = np.load(ckf_path)
                if file_num == 0:
                    cks = ck
                if cks.ndim != ck.ndim:
                    print("Error in array dimensions!")
                    break
                if file_num != 0:
                    cks = np.concatenate((cks, ck))
                file_num += 1

                if iter_num == iteration:
                    print(epoch_num)
                    print(sorted_files_by_time)
                    # Normalization
                    _, min, max = __normalize(cks)
                    norm = np.array([epoch_num, min, max])
                    np.save(assemble_path, cks)
                    np.save(normalization_path, norm)

                pbar.update(1)

    return cks


def aggregate(model_directory):
    assemble_directorys = os.listdir(model_directory)
    norms = []
    norm_num = 0
    for dir in assemble_directorys:
        path = os.path.join(model_directory, dir)
        if os.path.isdir(path):
            assemble_files = os.listdir(path)
            for f in assemble_files:
                if f == 'normalization.npy':
                    norm_path = os.path.join(path, f)
                    norm = np.load(norm_path)
                    norm = norm.reshape(1, norm.size)
                    if norm_num == 0:
                        norms = norm
                    if norm_num != 0:
                        norms = np.concatenate((norms, norm))

                    norm_num += 1

    print(f"norms shape is {norms.shape}")
    print(norms)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(norms)
    min = scaler.data_min_[1]
    max = scaler.data_max_[2]
    print(f"min is {min}, max is {max}")

    min_max_path = os.path.join(model_directory, "min_max.npy")
    min_max = np.array([min, max])
    min_max = min_max.reshape(1, min_max.size)
    np.save(min_max_path, min_max)
    return min, max


def distributed_normalization(model_directory, layer_name, min, max):
    assemble_directorys = os.listdir(model_directory)
    norms = []
    with tqdm(total=len(assemble_directorys), desc="Distributed Normalization") as pbar:
        for dir in assemble_directorys:
            path = os.path.join(model_directory, dir)
            if os.path.isdir(path):
                assemble_files = os.listdir(path)
                for f in assemble_files:
                    if f == (layer_name + '.npy'):
                        ck_path = os.path.join(path, f)
                        ck = np.load(ck_path)
                        ck = (ck - min) / (max - min)

                        ck_norm_path = os.path.join(path, layer_name + '_norm.npy')
                        np.save(ck_norm_path, ck)
                pbar.update(1)


def save_loss(loss, epoch, iter, lossf_path):
    loss_arr = np.array([loss, epoch, iter])
    loss_arr = loss_arr.reshape(1, loss_arr.size)

    if os.path.exists(lossf_path):
        loss_history = np.load(lossf_path)
        print(f'already stored {loss_history.shape[0]} loss values')
        loss_combine = np.concatenate((loss_history, loss_arr))
    else:
        loss_combine = loss_arr
    np.save(lossf_path, loss_combine)
    return loss, epoch, iter


def convert_npy_to_csv(npyf_path):
    npyf = np.load(npyf_path)
    df = pd.DataFrame(npyf)

    directory, npy_n = os.path.split(npyf_path)
    fname_array = npy_n.split('.')[:-1]
    format = npy_n.split('.')[-1]
    print(fname_array)
    fname = ".".join(fname_array)
    csv_n = fname + '.csv'
    csvf_path = os.path.join(directory, csv_n)

    df.to_csv(csvf_path, index=False)


def traverse(path):
    with open(path, 'r', newline='') as f:
        lines = f.readlines()
    num_lines = len(lines)
    print(f"Number of lines in CSV file: {num_lines}")
    return num_lines


def clean_npy(path):
    dataset = np.load(path)
    max = np.max(dataset)
    min = np.min(dataset)
    mean = np.mean(dataset)
    std = np.std(dataset)
    print(max)  # 0.0979154
    print(min)  # -0.10808951
    print(mean)  # -1.4533376e-05
    print(std)  # 0.020129047

    dataset = (dataset - min) / (max - min)
    output_file = '/CPSR/GPT/ck/module.layer.12.h.mlp.c_fc.weight_norm.npy'
    np.save(output_file, dataset)


def split_temporal():
    path = '/CPSR/GPT/ck/1/module.layer.12.h.mlp.c_fc.weight.npy'
    samples = np.load(path)
    split_path = '/CPSR/GPT/ck/1/module.layer.12.h.mlp.c_fc.weight_0_60.npy'
    split_samples = samples[0:60]
    print(f"split_samples shape is {split_samples.shape}")
    np.save(split_path, split_samples)


def split_spatial(norm_path):
    samples = np.load(norm_path)
    dir_name = os.path.dirname(norm_path)
    file_name = os.path.basename(norm_path)
    file_name = file_name.split('.npy')[0] + "_2dim.npy"
    split_path = os.path.join(dir_name, file_name)

    l, c = samples.shape
    split_samples = np.empty((l * 1024, 4096))

    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            split_samples[i * 1024 + (j // 4096), j % 4096] = samples[i, j]

    print(f"split_samples shape is {split_samples.shape}")
    np.save(split_path, split_samples)


def distributed_split_spatial():
    split_dir = os.path.join(global_ck_directory, "assemble", "layer.12.h.mlp.c_fc.weight")
    split_dirs = os.listdir(split_dir)

    with tqdm(total=len(split_dirs), desc="Splitting") as pbar:
        for dir in split_dirs:
            dir_ = os.path.join(split_dir, dir)
            if os.path.isdir(dir_):
                split_path = os.path.join(dir_, "layer.12.h.mlp.c_fc.weight_norm.npy")
                split_spatial(split_path)
                pbar.update(1)


def recovery(origin_recovery_path):
    predict_path = '/CPSR/Predict/predict_parameter.npy'
    origin_path = '/CPSR/GPT/checkpoint/1_0.pth'
    ck = torch.load(origin_path)

    predict_np = np.load(predict_path)  # (1024,4096)
    predict_tensor = torch.tensor(predict_np)
    with torch.no_grad():
        ck['model_state_dict']['layer.12.h.mlp.c_fc.weight'][:] = predict_tensor

    origin_name = os.path.basename(origin_path)
    predict_dir = os.path.dirname(predict_path)
    origin_predict_path = os.path.join(predict_dir, "predict_model_" + origin_name)
    torch.save(ck, origin_predict_path)

    origin_np = np.load(origin_recovery_path)
    origin_tensor = torch.tensor(origin_np)
    with torch.no_grad():
        ck['model_state_dict']['layer.12.h.mlp.c_fc.weight'][:] = origin_tensor
    origin_origin_path = os.path.join(predict_dir, "origin_model_" + origin_name)
    torch.save(ck, origin_origin_path)


def integration_loss():
    origin_path = '/CPSR/GPT/loss/loss_origin.npy'
    origin_continue_path = '/CPSR/GPT/loss/loss_origin_continue.npy'
    origin = np.load(origin_path)
    origin_continue = np.load(origin_continue_path)

    origin = np.concatenate((origin, origin_continue))
    np.save(origin_path, origin)
    print(origin)
    print(origin.shape)


def cut():
    origin_path = '/CPSR/GPT/checkpoint/1_0.pth'
    ck = torch.load(origin_path)
    model = ck['model_state_dict']
    model = ck['model_state_dict']['module.layer.12.h.mlp.c_fc.weight']

    new_path = '/CPSR/Predict/result/module.layer.12.h.mlp.c_fc.weight.pth'
    torch.save(model, new_path)


def predict_process():
    assemble("/CPSR/GPT/ck/layer.12.h.mlp.c_fc.weight", 2380)  # Takes ~10 minutes
    model_dir = "/CPSR/GPT/ck/assemble/layer.12.h.mlp.c_fc.weight"
    layer_n = 'layer.12.h.mlp.c_fc.weight'
    min, max = aggregate(model_dir)
    distributed_normalization(model_dir, layer_n, min, max)  # Takes ~54 seconds


def compare_origin_predict(origin_path, predict_path):
    origin_loss = np.load(origin_path)
    predict_loss = np.load(predict_path)

    if len(origin_loss) != len(predict_loss):
        print("Origin length differs from predict length!")
    else:
        length = len(origin_loss)
    difference = []
    for i in range(length):
        difference.append(origin_loss[i][0] - predict_loss[i][0])
        if difference[i] > 0:
            print(difference[i])


def extract_number(folder_name):
    # Extract numeric part from folder name
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else float('inf')  # Put non-numeric names last


def sort_folders(directory):
    # Get all subdirectories in the folder
    all_items = os.listdir(directory)
    folders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]

    # Sort by numeric value
    sorted_folders = sorted(folders, key=extract_number)
    return sorted_folders


def link(link_dir, model_name):
    if not os.path.exists(link_dir):
        print("The checkpoints directory does not exist.")
        return False

    sorted_ck_folders = sort_folders(link_dir)
    for ck_folder in sorted_ck_folders:
        print(ck_folder)
    with tqdm(total=len(sorted_ck_folders), desc="Concatenating in window") as pbar:
        for i in range(len(sorted_ck_folders)):
            ck_path = os.path.join(link_dir, sorted_ck_folders[i], model_name + "_norm.npy")  # Already normalized
            ck = np.load(ck_path)
            if i == 0:
                cks = ck
            if i != 0:
                cks = np.concatenate((cks, ck))
            pbar.update(1)

    save_path = os.path.join(link_dir, "predict_dataset.npy")
    np.save(save_path, cks)


def compare_mix(predict_path, origin_path, alpha, layer_n):
    predict_layer = torch.load(predict_path)['model_state_dict'][layer_n]
    origin = torch.load(origin_path)
    origin_layer = origin['model_state_dict'][layer_n]

    _alpha = 1 - alpha
    threshold = torch.quantile(origin_layer, _alpha)
    mask = origin_layer > threshold
    result = torch.where(mask, origin_layer, predict_layer)
    origin['model_state_dict'][layer_n][:] = result

    dir = os.path.dirname(predict_path)
    fname = os.path.basename(predict_path)
    fname_notype = fname.split('.')[0]
    ftype = fname.split('.')[1]
    mix_predict_path = os.path.join(dir, fname_notype + '_mix.' + ftype)
    torch.save(origin, mix_predict_path)


def invert_normalization(predict_path, origin_path, min_max_path):
    predict = np.load(predict_path)
    predict = predict.astype(np.float32)
    origin = np.load(origin_path)
    origin = origin[0:1024, :]

    min_max = np.load(min_max_path)
    min_max = min_max.astype(float)
    min, max = min_max[1], min_max[2]

    predict = predict * (max - min) + min
    origin = origin * (max - min) + min

    dir = os.path.dirname(predict_path)
    p_path = os.path.join(dir, "predict_parameter.npy")
    o_path = os.path.join(dir, "origin_parameter.npy")
    np.save(p_path, predict)
    np.save(o_path, origin)


if __name__ == '__main__':
    csv_path = '/CPSR/Diffusion-TS-main/Data/datasets/module.layer.12.h.mlp.c_fc.weight.csv'
    npy_path = '/CPSR/GPT/ck/1/module.layer.12.h.mlp.c_fc.weight.npy'

    # Example usage:
    # predict_process()
    # recovery("/CPSR/Predict/origin_parameter.npy")
