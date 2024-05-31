import numpy as np
tau_relax = 27
delta_tau = 5
P = 23
N = 128  # Number of sub-carriers
Lc = 32  # Length of cyclic prefix
Nu = N + Lc
M = Nu + N
epsilon = 0.1  # Example carrier frequency offset (CFO)
eta = np.random.uniform(0.01, 0.5)

def calculate_channel_gain_multi_path():
    # The direct path tau_p = 0 
    tau_p = [0]

    # Generate remaining delays tau_p using uniform distribution, excluding zero, allowing repeats
    remaining_tau_p = np.random.randint(1, tau_relax - delta_tau + 1, P - 1)

    # Combine and sort 
    tau_p = np.concatenate((tau_p, remaining_tau_p))
    tau_p = np.sort(tau_p)

    # Calculate amplitude gains alpha_p of h_tau_p
    alpha_p = np.exp(-eta * tau_p / 2)

    # Generate random phases phi_p
    phi_p = np.random.uniform(0, 2 * np.pi, P)

    # Calculate complex gains h_tau_p
    h_tau_p = alpha_p * np.exp(1j * phi_p)

    return (tau_p, h_tau_p)

def generate_transmitted_signal():
    # Step 1: Generate the data symbols d_k
    d_k = np.random.randn(N) + 1j * np.random.randn(N)  # Generate random complex symbols

    # Step 2: Calculate the time-domain samples s_n using IDFT
    s_n = np.zeros(N, dtype=complex)

    for n in range(N):
        s_n[n] = (1/N) * np.sum(d_k * np.exp(1j * 2 * np.pi * np.arange(N) * n / N))

    # Step 3: Append the cyclic prefix (CP)
    cp = s_n[-Lc:]  # Last Lc samples of s_n
    s_cp = np.concatenate((cp, s_n))  # Append CP to the beginning of s_n
    return s_cp


def generate_received_signal(tau_p, h_tau_p, s_cp):
    # Generate time offset
    tilde_tau = np.random.randint(0, N)

    # Generate M-length received signal
    r_n = np.zeros(M, dtype=complex)
    w_n = np.random.randn(M) + 1j * np.random.randn(M)  

    for n in range(M):
        r_n[n] = np.sum([h_tau_p[p] * s_cp[(n - tilde_tau - tau_p[p]) % Nu] * 
                        np.exp(1j * 2 * np.pi * epsilon * (n - tilde_tau) / N) 
                        for p in range(P)]) + w_n[n]

    return (tilde_tau, r_n)


dataset_y = []
dataset_t = []

N_train = 1e5
for i in range(N_train):
    (tau_p, h_tau_p) = calculate_channel_gain_multi_path()
    s_cp = generate_transmitted_signal()
    (tilde_tau, r_n) = generate_received_signal(tau_p, h_tau_p, s_cp)
    # Extract real and imaginary parts and interleave them
    y_i = np.zeros(2 * M)
    y_i[0::2] = np.real(r_n)  # Real parts at even indices
    y_i[1::2] = np.imag(r_n)  # Imaginary parts at odd indices
    y_i = y_i.reshape(2 * M, 1)

    # Construct true timing offsets t_i as one-hot encoded vectors
    t_i = np.zeros(Nu, dtype=int)

    index = int(np.floor(tilde_tau + 0.5 * (Lc + tau_relax)))
    t_i[index] = 1

    dataset_y.append(y_i)
    dataset_t.append(t_i)

dataset_y = np.array(dataset_y)
dataset_t = np.array(dataset_t)

print("Input dataset y (first example):")
print(dataset_y.shape)
print("\nData label t (first example):")
print(dataset_t.shape)