from dataclasses import dataclass

@dataclass
class NetworkConfig:
    n_sub_carriers: int = 128
    length_cyclic_prefix: int = 32
    n_multi_path: int = 23
    tau_relax: int = 27 # the relaxed restriction for propagation delay
    delta_tau: int = 5 # the delay fluctuation

@dataclass
class Config: 
    nbits: int = 1
    msg_length: int = 100800 # 100800 
    nfft: int = 64 #16
    nsymbol: int = 7
    npilot: int = 8 #2
    nguard: int = 8 #2
    SNR: float = 3.0
    max_TO: int = 20 # max timing offset
    min_TO: int = 11 # min timing offset
    ofdm: bool = True
    cp: bool = True
    longcp: bool = True
    batch_size: int = 512
    max_epoch_num: int = 5000
    init_learning: float = 0.001
    early_stop: int = 400
    save_dir: str = './output/'
    channel: str = 'EPA'
    token: str = 'OFDM'
    pilot: str = 'lte'




