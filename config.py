from dataclasses import dataclass

@dataclass
class NetworkConfig:
    n_sub_carriers: int = 128
    length_cyclic_prefix: int = 32
    n_multi_path: int = 23
    tau_relax: int = 27 # the relaxed restriction for propagation delay
    delta_tau: int = 5 # the delay fluctuation