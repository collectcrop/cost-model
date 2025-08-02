import struct
import numpy as np

def read_keys_from_binary(filename):
    keys = []
    record_size = 16  # 8 bytes for key + 8 bytes for data
    with open(filename, 'rb') as f:
        while chunk := f.read(record_size):
            key, _ = struct.unpack('QQ', chunk)  # Q = unsigned long long (8 bytes)
            keys.append(key)
    return np.array(keys, dtype=np.uint64)

def compute_kappa(keys):
    keys = np.sort(keys)  
    gaps = np.diff(keys)

    mu = np.mean(gaps)
    sigma2 = np.var(gaps, ddof=0)  # 计算方差
    kappa = mu ** 2 / sigma2 if sigma2 > 0 else float('inf')

    print(f"μ (mean gap): {mu}")
    print(f"σ² (gap variance): {sigma2}")
    print(f"κ = μ² / σ²: {kappa}")

    return mu, sigma2, kappa

def main():
    keys = read_keys_from_binary("pgm_test_file.bin")
    mu, sigma2, kappa = compute_kappa(keys)
    print(f"μ (mean gap): {mu}, sigma2 (gap variance): {sigma2}, kappa: {kappa}")
    
if __name__ == "__main__":
    main()