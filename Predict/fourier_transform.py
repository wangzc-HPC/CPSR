import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import gc
import time
import os
import pynvml
import threading
from threading import Thread
import matplotlib.pyplot as plt

def save_data(path, channel, window=5):
    # Load signal data
    signal = np.load(path)
    print("signal shape : ", signal.shape)

    feature = signal.shape[1]
    signal_num = signal.shape[0] // channel
    signal = signal.reshape(signal_num, channel * feature)  # (239, 1024 * 4096)
    print("signal shape : ", signal.shape)
    sample_num = max(signal_num - window + 1, 0)

    # Split into sliding windows
    data = np.zeros((sample_num, window, channel * feature))
    for i in range(sample_num):
        start = i
        end = i + window
        data[i, :, :] = signal[start:end, :]
    data = np.transpose(data, (0, 2, 1))
    print("data shape : ", data.shape)
    # np.save('./fft_binary/data.npy', data)
    data.tofile('./fft_binary/data.dat')

def save_fourier_real(window=5):
    num = 235
    feature = 4096 * 1024
    # Step 1: perform Fourier transform on each signal time series to obtain frequency-domain data
    data = np.memmap('./fft_binary/data.dat', dtype='float64', mode='r', shape=(num, feature, window))
    fft_signals = np.fft.fft(data, axis=2)  # takes ~70s
    print("fft_signals shape : ", fft_signals.shape)
    # Save real and imaginary parts separately (float64)
    real_part = fft_signals.real.astype(np.float64)
    # imag_part = fft_signals.imag.astype(np.float64)
    real_part.tofile('./fft_binary/fft_real_part.dat')
    # imag_part.tofile('./fft_binary/fft_imag_part.dat')
    # fft_signals.tofile('./fft_binary/fft_signals.dat')

def save_fourier_imag(window=5):
    num = 235
    feature = 4096 * 1024
    # Step 1: perform Fourier transform on each signal time series to obtain frequency-domain data
    data = np.memmap('./fft_binary/data.dat', dtype='float64', mode='r', shape=(num, feature, window))
    fft_signals = np.fft.fft(data, axis=2)
    print("fft_signals shape : ", fft_signals.shape)
    # Save real and imaginary parts separately (float64)
    # real_part = fft_signals.real.astype(np.float64)
    imag_part = fft_signals.imag.astype(np.float64)
    # real_part.tofile('./fft_binary/fft_real_part.dat')
    imag_part.tofile('./fft_binary/fft_imag_part.dat')

def frequency_division(channel, feature, n_head, window=5):
    fft_real_part = np.memmap('./fft_binary/fft_real_part.dat', dtype='float64', mode='r+', shape=(235, 1024 * 4096, window))
    fft_imag_part = np.memmap('./fft_binary/fft_imag_part.dat', dtype='float64', mode='r+', shape=(235, 1024 * 4096, window))
    print("fft_real_part shape : ", fft_real_part.shape)
    print("fft_imag_part shape : ", fft_imag_part.shape)

    # print("fft_real_part NaN : ", np.isnan(fft_real_part).any())  # check for NaN
    # print("fft_real_part Inf : ", np.isinf(fft_real_part).any())  # check for Inf
    # print("fft_imag_part NaN : ", np.isnan(fft_imag_part).any())  # check for NaN
    # print("fft_imag_part Inf : ", np.isinf(fft_imag_part).any())  # check for Inf

    # Get frequency axis coordinates
    frequencies = np.fft.fftfreq(window, d=1)  # sampling interval is 1
    print("frequencies", frequencies)

    # Step 2: extract low-, mid-, and high-frequency components
    low_freq_idx = np.abs(frequencies) < 0.1
    high_freq_idx = np.abs(frequencies) > 0.25
    middle_freq_idx = ~low_freq_idx & ~high_freq_idx
    # np.save("middle_freq_idx.npy", middle_freq_idx)
    print("low_freq_idx", low_freq_idx)
    print("high_freq_idx", high_freq_idx)
    print("middle_freq_idx", middle_freq_idx)

    low_real_component = fft_real_part[:, :, low_freq_idx]
    print("low_real_component shape: ", low_real_component.shape)
    middle_real_component = fft_real_part[:, :, middle_freq_idx]
    print(middle_real_component)
    middle_imag_component = fft_imag_part[:, :, middle_freq_idx]
    print("middle_real_component shape: ", middle_real_component.shape)

    # Normalize mid-frequency data (Min-Max scaling to [0, 1])
    middle_real_min = middle_real_component.min()
    middle_real_max = middle_real_component.max()
    middle_real_component_normalized = (middle_real_component - middle_real_min) / (middle_real_max - middle_real_min)

    middle_imag_min = middle_imag_component.min()
    middle_imag_max = middle_imag_component.max()
    middle_imag_component_normalized = (middle_imag_component - middle_imag_min) / (middle_imag_max - middle_imag_min)

    # Save normalization parameters
    normalization_params = {
        "middle_real_min": middle_real_min,
        "middle_real_max": middle_real_max,
        "middle_imag_min": middle_imag_min,
        "middle_imag_max": middle_imag_max,
    }
    print("middle_real_min: ", middle_real_min)
    print("middle_real_max: ", middle_real_max)
    print("middle_imag_min: ", middle_imag_min)
    print("middle_imag_max: ", middle_imag_max)
    # np.save("normalization_params.npy", normalization_params)

    # Original shape (239, 1024*4096, 2), now reshape to (239*1024, 2, 4096)
    print("middle_real_component_normalized.shape is {}".format(middle_real_component_normalized.shape))
    n_samples = middle_real_component_normalized.shape[0]
    middle_window = middle_real_component_normalized.shape[2]
    feature_n = middle_real_component_normalized.shape[1]
    n_feature = feature_n // channel
    n_channel = channel
    middle_real_component_normalized = middle_real_component_normalized.reshape(n_samples, n_channel, n_feature, middle_window)
    middle_real_component_normalized = np.transpose(middle_real_component_normalized, (0, 1, 3, 2))
    middle_real_component_normalized = middle_real_component_normalized.reshape(n_samples * n_channel, middle_window, n_feature)
    print("flag1")

    middle_imag_component_normalized = middle_imag_component_normalized.reshape(n_samples, n_channel, n_feature, middle_window)
    middle_imag_component_normalized = np.transpose(middle_imag_component_normalized, (0, 1, 3, 2))
    middle_imag_component_normalized = middle_imag_component_normalized.reshape(n_samples * n_channel, middle_window, n_feature)
    print("flag2")

    return middle_real_component_normalized, middle_imag_component_normalized, middle_freq_idx, normalization_params

def fourier(path, window, channel, feature, n_head):
    # save_data(path, channel, window=window)
    # save_fourier_real(window)
    # save_fourier_imag(window)
    middle_real_component_normalized, middle_imag_component_normalized, middle_freq_idx, normalization_params = frequency_division(channel, feature, n_head, window)
    return middle_real_component_normalized, middle_imag_component_normalized, middle_freq_idx, normalization_params

def inverse_normalization(data_real_normalized, data_imag_normalized, normalization_params):
    """
    Restore normalized data to original range
    """
    middle_real_min = normalization_params["middle_real_min"]
    middle_real_max = normalization_params["middle_real_max"]
    middle_imag_min = normalization_params["middle_imag_min"]
    middle_imag_max = normalization_params["middle_imag_max"]
    data_real_original = data_real_normalized * (middle_real_max - middle_real_min) + middle_real_min
    data_imag_original = data_imag_normalized * (middle_imag_max - middle_imag_min) + middle_imag_min
    return data_real_original, data_imag_original

# Build custom Dataset and DataLoader
class TrainedDataset(Data.Dataset):
    def __init__(self, signals, time_step=1024):
        self.data = signals[:-time_step]  # first N‑1024 timesteps
        self.target = signals[time_step:]  # use from timestep N+1024 as target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

class TestDataset(Data.Dataset):
    def __init__(self, data):
        """
        data: test data, shape (1024, 2, 4096)
        """
        self.data = data

    def __len__(self):
        return len(self.data)  # return number of samples

    def __getitem__(self, idx):
        # return one sample
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)  # cast to Tensor

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(TransformerTimeSeries, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)  # project features into hidden_dim
        self.positional_encoding = nn.Parameter(torch.zeros(1, 2, hidden_dim))  # window size = 2
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, output_dim)  # map hidden_dim back to input_dim

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim) -> (2, 4096)
        """
        x = self.input_projection(x) + self.positional_encoding  # add positional encoding
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, input_dim) -> (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # back to (batch_size, seq_len, hidden_dim)
        x = self.output_projection(x)  # projection to output_dim
        return x  # (batch_size, seq_len, input_dim)

def create_dataloader(normalized_data, channel, n_head):
    signals_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    dataset = TrainedDataset(signals_tensor, time_step=channel)
    dataloader = Data.DataLoader(dataset, batch_size=32 * n_head, shuffle=True)
    return dataloader, signals_tensor

def test(middle_signals_tensor, model, device, channel, feature, flag):
    # Suppose test data is shape (1024, 2, 4096)
    test_data = middle_signals_tensor[-channel:]
    test_window = test_data.shape[1]
    test_dataset = TestDataset(test_data)

    batch_size = 32
    test_dataloader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    num_tests = len(test_dataset)
    predict_results = np.zeros((num_tests, test_window, feature), dtype=np.float32)

    model.eval()
    start_idx = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            predictions = model(batch)
            batch_size = predictions.size(0)
            predict_results[start_idx:start_idx + batch_size] = predictions.cpu().numpy()
            start_idx += batch_size

    predict_results = np.transpose(predict_results, (1, 0, 2))
    predict_results = predict_results.reshape(1, test_window, channel*feature)
    predict_results = np.transpose(predict_results, (0,2,1))
    print("Shape before inverse normalization:", predict_results.shape)
    if flag == "real":
        print("predict_real_results shape : ", predict_results.shape)
        np.save("predict_real_results.npy", predict_results)
    if flag == "imag":
        print("predict_imag_results shape : ", predict_results.shape)
        np.save("predict_imag_results.npy", predict_results)
    return predict_results

class GPUMonitor:
    def __init__(self, device=0, interval=0.01):
        self.device = torch.device(f'cuda:{device}')
        self.interval = interval
        self.mem_usage = []
        self.timestamps = []
        self.running = False

    def start(self):
        self.running = True
        self.thread = Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        start_time = time.time()
        while self.running:
            mem = torch.cuda.memory_allocated(self.device) / 1024**2
            self.mem_usage.append(mem)
            self.timestamps.append(time.time() - start_time)
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.thread.join()

    def save(self, filename_npz='/lihongliang/wangzc/Paper-2/figure/gpu_memory_usage.npz'):
        np.savez(filename_npz,
                 timestamps=np.array(self.timestamps),
                 memory_usage_MB=np.array(self.mem_usage))

def train(path):
    # Create dataset and dataloader
    window = 5
    n_head = 16
    channel = 1024
    feature = 4096
    middle_real_component_normalized, middle_imag_component_normalized, middle_freq_idx, normalization_params = fourier(path, window, channel, feature, n_head)
    print("middle_real_component_normalized", middle_real_component_normalized)
    print("middle_real_component_normalized shape", middle_real_component_normalized.shape)
    print("middle_imag_component_normalized", middle_imag_component_normalized)
    print("middle_imag_component_normalized shape", middle_imag_component_normalized.shape)

    start = time.time()
    print("start time : ", start)

    real_dataloader, middle_real_signals_tensor = create_dataloader(middle_real_component_normalized, channel, n_head)
    imag_dataloader, middle_imag_signals_tensor = create_dataloader(middle_real_component_normalized, channel, n_head)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTimeSeries(input_dim=min(feature, channel)//n_head, hidden_dim=128, num_heads=4, num_layers=3, output_dim=min(feature, channel)//n_head).to(device)
    model = TransformerTimeSeries(input_dim=feature, hidden_dim=128, num_heads=4, num_layers=3, output_dim=feature).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    loss_path = "/lihongliang/wangzc/Predict/loss.npy"
    loss_list = []

    # === Start monitoring ===
    monitor = GPUMonitor(interval=0.01)
    monitor.start()

    epochs = 4
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        real_dataloader_iter = iter(real_dataloader)
        imag_dataloader_iter = iter(imag_dataloader)
        union_iter = len(real_dataloader) + len(imag_dataloader)
        print(f'union_iter number is {union_iter}')

        for index in range(union_iter):
            if index % 2 == 0:
                inputs, targets = next(real_dataloader_iter)
            else:
                inputs, targets = next(imag_dataloader_iter)

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if index % 2 == 0:
                print(loss.item())
                loss_list.append(loss.item())
                if loss.item() < 0.006:
                    print(f"index is {index}")

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(real_dataloader):.4f}')

    # === Stop monitoring ===
    monitor.stop()
    monitor.save('/lihongliang/wangzc/Paper-2/figure/gpu_memory_usage_4096.npz')

def combine(predict_results_real, predict_results_imag, normalization_params, middle_freq_idx):
    # fft_signals = np.memmap('./fft_binary/fft_signals_complex128.dat', dtype='complex128', mode='r+', shape=(235, 1024 * 4096, window))
    num = 235
    window = 5
    feature = 4096 * 1024
    data = np.memmap('./fft_binary/data.dat', dtype='float64', mode='r', shape=(num, feature, window))
    fft_signals = np.fft.fft(data, axis=2)
    del data
    gc.collect()
    # Inverse normalization
    predict_results_real, predict_results_imag = inverse_normalization(predict_results_real, predict_results_imag, normalization_params)

    # Combine real and imaginary parts
    predict_results = predict_results_real + 1j * predict_results_imag
    print("predict_results shape : ", predict_results.shape)  # should be (1, 4194304, 2)
    middle_window = np.sum(middle_freq_idx)
    predict_results = predict_results.reshape(feature, middle_window)
    predict_results = np.transpose(predict_results, (1, 0))
    print("predict_results shape : ", predict_results.shape)
    np.save("predict_results.npy", predict_results)

    # Merge predicted mid-frequency part
    # fft_signals = fft_signals.copy()
    fft_signals[-1, :, middle_freq_idx] = predict_results
    fft_signals = fft_signals[-1, :, :]
    print("fft_signals shape :", fft_signals.shape)

    # Use inverse Fourier transform (IFFT) to recover time-domain signal
    predicted_signal_complex = np.fft.ifft(fft_signals, axis=1)
    print("predicted_signal_complex shape : ", predicted_signal_complex.shape)
    predicted_signal_complex = predicted_signal_complex[:, -1]
    print("predicted_signal_complex shape : ", predicted_signal_complex.shape)
    np.save("predicted_signal_complex.npy", predicted_signal_complex)

    predicted_signal_real = predicted_signal_complex.real
    predicted_signal_real = predicted_signal_real.reshape(1024, 4096)
    np.save("predicted_signal_real.npy", predicted_signal_real)

# Main call
path = "/lihongliang/wangzc/GPT-2/ck/assemble/layer.12.h.mlp.c_fc.weight/0/layer.12.h.mlp.c_fc.weight_norm.npy"
middle_freq_idx, normalization_params, predict_results_real, predict_results_imag = train(path)

# window = 5
channel = 1024
# feature = 4096
# combination commented out as per original
# combine(predict_results_real, predict_results_imag, normalization_params, middle_freq_idx)
# save_fourier()
# folder_path = "/lihongliang/wangzc/GPT/dp/origin_data"
# save_data(folder_path, channel, window=5)
