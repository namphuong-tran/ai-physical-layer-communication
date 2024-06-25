import numpy as np
from radio import *
from sklearn.model_selection import train_test_split
import os
from config import Config
from ofdm import *


def bit_source(nbits, frame_size, msg_length):
    '''
    Generate uniform random m_order one hot symbols
    :param frame_size: Equivalent to FFT size in OFDM
    :param msg_length: number of frames
    :return: bits
    '''
    # nbits = int(np.log2(m_order))
    bits = np.random.randint(0, 2, (int(msg_length), int(frame_size), int(nbits)))
    return bits


def generate_and_save_chunk():
    config =  Config
    nbits = config.nbits # BPSK: 2, QPSK, 4, 16QAM: 16
    ofdmobj = ofdm_tx(config)
    frame_size = ofdmobj.frame_size
    frame_cnt = config.msg_length//config.nsymbol
    fading = rayleigh_chan_lte(config, ofdmobj.Fs)
    train_ys = bit_source(nbits, frame_size, frame_cnt)
    snr_seq = np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    train_snr = 3.0 + np.repeat(snr_seq, frame_cnt//8, axis=0)
    iq_tx_cmpx, train_xs, iq_pilot_tx = ofdmobj.ofdm_tx_frame_np(train_ys)
    train_xs, labels, chan_xs = fading.run(iq_tx_cmpx)
    train_xs, pwr_noise_avg = AWGN_channel_np(train_xs, train_snr)
    train_xs = train_xs.reshape(train_xs.shape[0], -1)  # Shape: [n_fr, n_sym * n_sc * 2]

    return train_xs, labels

# Function to save smaller files
def save_smaller_files(data, labels, data_points_per_file, output_dir, prefix):
    output_dir = output_dir + '/' + prefix
    os.makedirs(output_dir, exist_ok=True)
    num_files = int(np.ceil(len(data) / data_points_per_file))
    
    for i in range(num_files):
        start_idx = i * data_points_per_file
        end_idx = min((i + 1) * data_points_per_file, len(data))
        
        file_data = data[start_idx:end_idx]
        file_labels = labels[start_idx:end_idx]
        
        np.save(os.path.join(output_dir, f'{prefix}_data_{i}.npy'), file_data)
        np.save(os.path.join(output_dir, f'{prefix}_labels_{i}.npy'), file_labels)


data, labels = generate_and_save_chunk()
# Parameters
test_size = 0.2  # 20% test data
data_points_per_file = 10000
output_dir = 'output_data'

# Split into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, random_state=42)
print(train_data.shape)
# Save smaller files
save_smaller_files(train_data, train_labels, data_points_per_file, output_dir, 'train')
save_smaller_files(test_data, test_labels, data_points_per_file, output_dir, 'test')

print("Data has been split and saved into smaller files.")



